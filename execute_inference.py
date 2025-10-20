import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# ⚙️ 설정
# ============================================================
isQLora = True   # ✅ False로 바꾸면 LoRA 추론
base_model_id = "meta-llama/Llama-3.2-3B"
adapter_path = "./trained-llama3-qlora/checkpoint-1000" if isQLora else "./trained-llama3-lora/checkpoint-1000"

# ============================================================
# ✅ 1. 모델 로드
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

# 🚫 merge_and_unload() 절대 사용하지 않음 (QLoRA일 때 rounding 문제)
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()

# ============================================================
# ✅ 2. 토크나이저
# ============================================================
tok = AutoTokenizer.from_pretrained(base_model_id)
tok.pad_token = tok.eos_token
tok.padding_side = "right"

# ============================================================
# ✅ 3. 프롬프트
# ============================================================
prompt = "Answer like Thrall.\nTell me about who you are.\nAnswer: "

inputs = tok(prompt, return_tensors="pt").to(model.device)

# ============================================================
# ✅ 4. 생성
# ============================================================
with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=60,
        min_new_tokens=10,   # ✅ 최소 길이 확보
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
'''
with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
'''
decoded = tok.decode(out[0], skip_special_tokens=True)

print("===============================================")
print(f"🦙 Mode     : {'QLoRA (4bit)' if isQLora else 'LoRA (FP16)'}")
print("prompt : ", prompt)
print("answer : ", decoded[len(prompt):].strip())
print("===============================================")
