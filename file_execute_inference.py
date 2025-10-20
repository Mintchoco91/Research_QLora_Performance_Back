import json
import torch
from tqdm import tqdm
from bert_score import score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from statistics import mean
import numpy as np   # ✅ np.std 사용하므로 꼭 import 필요

# ============================================================
# ⚙️ 1. 설정
# ============================================================
weight = "1B"
isQLora = True
base_model_id = "meta-llama/Llama-3.2-" + weight
rank_val = 32
alpha_val = 64
model_dir = f"weight-{weight}-rank-{rank_val}-alpha-{alpha_val}-qlora-{isQLora}"
adapter_path = f"./result/{model_dir}/checkpoint-400"
test_file = "./train_data/final_test.jsonl"

# ============================================================
# ✅ 2. 토크나이저 로드
# ============================================================
tok = AutoTokenizer.from_pretrained(base_model_id)
tok.pad_token = tok.eos_token
tok.padding_side = "right"

# ============================================================
# ✅ 3. 모델 로드
# ============================================================
if isQLora:
    print("🔹 Loading QLoRA 4bit model...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map="auto"
    )
else:
    print("🔹 Loading LoRA FP16 model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

# PEFT 어댑터 로드
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()

# ============================================================
# ✅ 4. 테스트 데이터 불러오기
# ============================================================
test_data = []
with open(test_file, "r", encoding="utf-8") as f:
    for line in f:
        test_data.append(json.loads(line.strip()))

print(f"✅ Loaded {len(test_data)} samples from {test_file}")

# ============================================================
# ✅ 5. FBERT (BERTScore,  Persona Accuracy) 계산
# ============================================================
num_runs = 5
fbert_list, acc_list = [], []

for run in range(num_runs):
    print(f"🔁 Run {run+1}/{num_runs}")
    preds, refs = [], []

    for sample in tqdm(test_data, desc=f"Run {run+1} Inference"):
        prompt = f"{sample['instruction']}\n{sample['input']}\nAnswer: " if sample.get("input") else f"{sample['instruction']}\nAnswer: "
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                min_new_tokens=10,
                do_sample=True,          # ✅ 랜덤 샘플링 유지
                temperature=0.6,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=tok.eos_token_id,
            )
        decoded = tok.decode(out[0], skip_special_tokens=True)
        pred = decoded[len(prompt):].strip()
        preds.append(pred)
        refs.append(sample["output"].strip())

    # FBERT 측정
    P, R, F1 = score(preds, refs, lang="en")
    fb_f1 = F1.mean().item()
    fbert_scores = F1.tolist()
    persona_acc = sum(1 for s in fbert_scores if s >= 0.8) / len(fbert_scores)

    fbert_list.append(fb_f1)
    acc_list.append(persona_acc)

print("===============================================")
print(f"FBERT (mean ± std): {np.mean(fbert_list):.4f} ± {np.std(fbert_list):.4f}")
print(f"Persona Accuracy (mean ± std): {np.mean(acc_list)*100:.2f}% ± {np.std(acc_list)*100:.2f}%")
print("===============================================")
