import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    BitsAndBytesConfig, 
    EarlyStoppingCallback, 
    default_data_collator,
    DataCollatorWithPadding,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
import math
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from torch.optim import AdamW
import torch
import wandb
from torch.nn import CrossEntropyLoss
import time

from datasets import concatenate_datasets
import torch
print("is_available ? :  ", torch.cuda.is_available())
# ============================================================
# ✅ 실험 데이터
# ============================================================
idx = 5
weight = "1B"
model_id = "meta-llama/Llama-3.2-" + weight
isQLora = False  # ← LoRA로 바꾸려면 False로
rank_val = 2
alpha_val = 2
Scaling_factor = float(alpha_val/rank_val)


class VRAMLoggingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # GPU 메모리 사용량 측정 (MB 단위)
        real_vram_peak = torch.cuda.memory_allocated() / 1024**2
        reserve_vram_step = torch.cuda.max_memory_reserved() / 1024**2
        wandb.log({"Real_Vram_Step_MB": real_vram_peak, "step": state.global_step})
        wandb.log({"Reserve_Vram_Step_MB": reserve_vram_step, "step": state.global_step})

        


# =========================================
# ✅ Trainable Parameters 계산 함수
# =========================================
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    percent = 100 * trainable_params / all_params
    print(f"Trainable params: {trainable_params:,}")
    print(f"All params: {all_params:,}")
    print(f"Trainable%: {percent:.4f}%")
    return percent



# ============================================================
# ✅ 1. 데이터 전처리 함수
# ============================================================
def tokenize_func(batch):
    prompts = []
    texts = []

    for instruction, user_input, out in zip(
        batch["instruction"], batch["input"], batch["output"]
    ):
        prompt = f"""### Instruction:
{instruction}

### Input:
{user_input}

### Response:
"""
        prompts.append(prompt)
        texts.append(prompt + out + tokenizer.eos_token)

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=128,
        padding=True,
        add_special_tokens=False
    )

    labels = []
    for i in range(len(texts)):
        input_ids = tokenized["input_ids"][i]
        prompt_ids = tokenizer(prompts[i], add_special_tokens=False)["input_ids"]

        label = input_ids.copy()
        label[:len(prompt_ids)] = [-100] * len(prompt_ids)
        labels.append(label)

    tokenized["labels"] = labels

    return tokenized

# ============================================================
# ✅ Perplexity
# ============================================================
def evaluate_token_weighted_ppl(trainer, eval_dataset):
    model = trainer.model
    model.eval()
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="sum")  # 합계로 모음
    total_nll = 0.0
    total_tokens = 0

    dataloader = trainer.get_eval_dataloader(eval_dataset)
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask", None),
                            labels=batch["labels"])
            # outputs.loss는 배치 평균. 대신 로짓으로 직접 합계 NLL 계산
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = batch["labels"][:, 1:].contiguous()  # next-token loss
            # 평탄화
            shift_logits = logits.view(-1, logits.size(-1))
            shift_labels = labels.view(-1)
            nll = loss_fct(shift_logits, shift_labels)  # 합계 NLL
            valid = (shift_labels != -100).sum()

            total_nll += nll.item()
            total_tokens += valid.item()

    avg_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(avg_nll)
    return ppl, avg_nll, total_tokens

# ============================================================
# ✅ 2. 모델 및 토크나이저 로드
# ============================================================
#model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"isQLora Mode : ", str(isQLora))

wandb.init(
    project="llama3-qlora",
    name="idx-" + str(idx) +"-weight-"+ str(weight) + "-rank-" + str(rank_val) + "-alpha-" + str(alpha_val) + "-qlora-" + str(isQLora),
    config={"lr": 5e-5, "epochs": 8}
)

if isQLora:
    # ✅ QLoRA 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    optim_type = "adamw_bnb_8bit"

else:
    # ✅ 일반 LoRA 설정 (비양자화)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    optim_type = "adamw_torch"

output_dir_name = "../result/weight-"+ str(weight) + "-rank-" + str(rank_val) + "-alpha-" + str(alpha_val) + "-qlora-" + str(isQLora)


# ============================================================
# ✅ 3. LoRA 설정
# ============================================================
model = prepare_model_for_kbit_training(model)
#속도 절감
#model.gradient_checkpointing_enable()
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id

lora_config = LoraConfig(
    r=rank_val,
    lora_alpha=alpha_val,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 4️⃣ 파라미터 계산
trainable_percent = print_trainable_parameters(model)
model.print_trainable_parameters()


# ============================================================
# ✅ 4. 데이터셋 로드 및 토크나이징
# ============================================================
dataset = load_dataset(
    "json",
    data_files={
        "train": "../train_data/api_train.jsonl",
        #"train": "./train_data/custom_train.jsonl",
        "test": "../train_data/eval.jsonl"
    }
)

# ✅ 여기서 샘플 개수 줄이기 (sanity check용)
#dataset["train"] = dataset["train"].select(range(100))   # 훈련 데이터 8개만
#dataset["test"]  = dataset["test"].select(range(2))    # 평가 데이터 2개만
#dataset["train"] = concatenate_datasets([dataset["train"]] * 3)
dataset["train"] = dataset["train"].shuffle(seed=42)


tokenized_train = dataset["train"].map(
    tokenize_func,
    batched=True,
    remove_columns=dataset["train"].column_names
)

tokenized_eval = dataset["test"].map(
    tokenize_func,
    batched=True,
    remove_columns=dataset["test"].column_names
)


# ============================================================
# ✅ 5. Trainer 세팅
# ============================================================
training_args = TrainingArguments(
    # ⚙️ 기본 학습 설정
    per_device_train_batch_size = 1,       # 8GB VRAM 안전선
    gradient_accumulation_steps = 2,       # 실효 배치 4 → 안정적 수렴
    num_train_epochs = 5,                  # ✅ 3 → 2로 줄임 (과적합 방지)
    learning_rate = 3e-5,                  # ✅ 2e-4 → 1e-4로 완화 (loss 급락 억제)

    # 🔄 스케줄 및 최적화
    warmup_ratio = 0.05,                   # ✅ 0.03 → 0.05로 상승 (초반 안정성 강화)
    lr_scheduler_type = "cosine",          # cosine decay로 완만한 감소
    weight_decay = 0.05,                   # ✅ 0.01 → 0.05로 증가 (일반화 강화)
    max_grad_norm = 0.8,                   # ✅ 그래디언트 폭주 방지 강화
    optim = optim_type,                    # LoRA: adamw_torch / QLoRA: adamw_bnb_8bit

    # 🧩 정밀도 설정
    fp16 = False,                          # ✅ QLoRA overflow 방지
    bf16 = False,                          # RTX 30 시리즈 미지원

    # 🧾 로깅 & 저장
    logging_steps = 100,
    save_strategy = "epoch",               # 에폭 단위 저장
    eval_strategy = "epoch",               # 에폭 단위 평가
    save_total_limit = 3,                  # 최대 3개 체크포인트 유지
    load_best_model_at_end = True,         # 검증 손실 기준 복원
    logging_first_step = True,             # 첫 step부터 로깅

    # 🪣 출력 및 관리
    output_dir = output_dir_name,
    run_name = "llama3-finetune-compare",
    report_to = "wandb",
)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=training_args.learning_rate
)

# ✅ collator 변경 (labels 유지)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator= data_collator,
    optimizers=(optimizer, None),
    callbacks=[VRAMLoggingCallback()]
)


# ============================================================
# ✅ 6. 학습 실행
# ============================================================
start_time = time.time()    #소요시간 측정 시작
trainer.train()
end_time = time.time()      #소요시간 측정 끝

elapsed = end_time - start_time
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = int(elapsed % 60)
time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

wandb.log({
    "train_time_hms": time_str,  # 보기 좋은 HH:MM:SS 형식
    "train_time_hours": elapsed / 3600,  # 그래프용 숫자형 데이터
    "train_time_minutes": elapsed / 60   # 비교용
})

metrics = trainer.evaluate()
ppl = math.exp(metrics["eval_loss"])
print(metrics)


# 🔹 여기서 정확한 PPL 계산
token_weighted_ppl, avg_nll, ntoks = evaluate_token_weighted_ppl(trainer, tokenized_eval)
wandb.log({
    "Perplexity_token_weighted": token_weighted_ppl,
    "Eval_AvgNLL": avg_nll,
    "Eval_Tokens": ntoks
})
print(f"Token-weighted PPL: {token_weighted_ppl:.3f} (avg NLL: {avg_nll:.5f}, tokens: {ntoks})")

# Peak VRAM 기록
real_vram_peak = torch.cuda.memory_allocated() / 1024**2
reserve_vram_peak = torch.cuda.max_memory_reserved() / 1024**2
wandb.log({"Real_VRAM_Peak_MB": real_vram_peak})
wandb.log({"Reserve_VRAM_Peak_MB": reserve_vram_peak})

# ============================================================
# ✅ 7. 모델 저장
# ============================================================
trainer.model.save_pretrained("../llama-3.2-1b-lora")
tokenizer.save_pretrained("../llama-3.2-1b-lora")

print("Last checkpoint:", trainer.state.best_model_checkpoint)
print("Output directory:", training_args.output_dir)

# ============================================================
# ✅ 8. 실험 결과
# ============================================================

print("============================================================")
print(f"isQLora : {isQLora}")
print(f"rank_val : {rank_val}")
print(f"alpha_val : {alpha_val}")
print(f"Scaling_factor : {Scaling_factor}")
print(f"Trainable Params(%) : {print_trainable_parameters(model)}")


print(f"Training Time : {time_str}")
print(f"Perplexity: {token_weighted_ppl}")
print(f"Real_VRAM_Peak_MB: {real_vram_peak:.2f} MB")
print(f"Reserve_VRAM_Peak_MB: {reserve_vram_peak:.2f} MB")
print("============================================================")