import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# ⚙️ 모델 로드
# ============================================================
weight = "3B"
isQLora = True
base_model_id = f"meta-llama/Llama-3.2-{weight}"
rank_val = 4
alpha_val = 4
model_dir = f"weight-{weight}-rank-{rank_val}-alpha-{alpha_val}-qlora-{isQLora}"
adapter_path = f"../result/{model_dir}/checkpoint-400"

if isQLora:
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
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

model = PeftModel.from_pretrained(base, adapter_path)
model.eval()

tok = AutoTokenizer.from_pretrained(base_model_id)
tok.pad_token = tok.eos_token
tok.padding_side = "right"


# ============================================================
# ✅ 추론 함수
# ============================================================
def run_inference(
    user_input: str,
):
    instruction = "쓰랄의 소개에 대한 정보"
    prompt = f"{instruction}\n{user_input}\n답변: "
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
    answer = decoded[len(prompt):].strip()

    return {
        "mode": "QLoRA (4bit)" if isQLora else "LoRA (FP16)",
        "prompt": prompt,
        "answer": answer
    }
