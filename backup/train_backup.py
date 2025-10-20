import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    BitsAndBytesConfig, 
    EarlyStoppingCallback, 
    default_data_collator,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from torch.optim import AdamW
import torch
import wandb

# ============================================================
# ✅ 1. 데이터 전처리 함수
# ============================================================
def tokenize_func(batch):
    texts = []
    for instr, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
        prompt = f"{instr}\n{inp}\nAnswer: " if inp else f"{instr}\nAnswer: "
        text = prompt + out

        # Dynamic padding
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",   # ✅ 고정 패딩
            add_special_tokens=False
        )

        # Mask prompt tokens (-100)
        labels_ids = tokenized["input_ids"].copy()
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        labels_ids[:prompt_len] = [-100] * prompt_len
        tokenized["labels"] = labels_ids
        texts.append(tokenized)

    # Batchify
    return {k: [dic[k] for dic in texts] for k in texts[0]}


# ============================================================
# ✅ 2. 모델 및 토크나이저 로드
# ============================================================
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

isQLora = False  # ← LoRA로 바꾸려면 False로

wandb.init(
    project="llama3-qlora",
    name="llama3-run2",
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
    output_dir_name = "./trained-llama3-qlora"
else:
    # ✅ 일반 LoRA 설정 (비양자화)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    optim_type = "adamw_torch"
    output_dir_name = "./trained-llama3-lora"


# ============================================================
# ✅ 3. LoRA 설정
# ============================================================
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ============================================================
# ✅ 4. 데이터셋 로드 및 토크나이징
# ============================================================
dataset = load_dataset(
    "json",
    data_files={
        #"train": "./train_data/train.jsonl",
        "train": "./train_data/custom_train.jsonl",
        "test": "./train_data/eval.jsonl"
    }
)

# ✅ 여기서 샘플 개수 줄이기 (sanity check용)
dataset["train"] = dataset["train"].select(range(100))   # 훈련 데이터 8개만
dataset["test"]  = dataset["test"].select(range(2))    # 평가 데이터 2개만


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
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=5,
    learning_rate=1e-4,
    warmup_ratio=0.0,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    logging_steps=20,

    save_strategy="steps",
    save_steps=125,
    eval_strategy="steps",
    eval_steps=125,

    optim=optim_type,
    max_grad_norm=1.0,
    fp16=False,
    bf16=False,
    output_dir=output_dir_name,
    run_name="llama3-qlora-run2",
    report_to="wandb",
    save_total_limit=3,
    load_best_model_at_end=True,
    logging_first_step=True,
)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=training_args.learning_rate
)

# ✅ collator 변경 (labels 유지)
data_collator = default_data_collator

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, None),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)


# ============================================================
# ✅ 6. 학습 실행
# ============================================================
trainer.train()
metrics = trainer.evaluate()
print(metrics)

# ============================================================
# ✅ 7. 모델 저장
# ============================================================
trainer.model.save_pretrained("./llama-3.2-1b-lora")
tokenizer.save_pretrained("./llama-3.2-1b-lora")

print("Last checkpoint:", trainer.state.best_model_checkpoint)
print("Output directory:", training_args.output_dir)
