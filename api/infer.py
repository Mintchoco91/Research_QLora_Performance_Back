import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

# ============================================================
# ⚙️ 모델 로드
# ============================================================
weight = "3B"
isQLora = True
base_model_id = f"meta-llama/Llama-3.2-{weight}"
rank_val = 32
alpha_val = 64
model_dir = f"weight-{weight}-rank-{rank_val}-alpha-{alpha_val}-qlora-{isQLora}"
adapter_path = f"../result/{model_dir}/checkpoint-260"

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

def keep_korean(text):
    allowed = set(" .,!?~…")
    return ''.join([ch for ch in text if '가' <= ch <= '힣' or ch in allowed])

def contains_non_korean(text):
    for ch in text:
        if not (ord('가') <= ord(ch) <= ord('힣') or ch in " .,!?"):
            return True
    return False

# ============================================================
# ✅ 추론 함수
# ============================================================
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

identity_samples = ["너 누구야", "정체가 뭐야", "이름이 뭐야"]
goodbye_samples = ["잘 가", "다음에 보자", "이만 갈게"]

def classify_intent(text):
    emb = sbert_model.encode(text, convert_to_tensor=True)

    id_score = util.cos_sim(emb, sbert_model.encode(identity_samples, convert_to_tensor=True)).max()
    gb_score = util.cos_sim(emb, sbert_model.encode(goodbye_samples, convert_to_tensor=True)).max()

    return "identity" if id_score > gb_score else "goodbye"


def make_system_hint(intent: str):
    if intent == "identity":
        return "사용자는 너의 정체를 궁금해하고 있다."
    elif intent == "goodbye":
        return "사용자는 작별 인사를 하고 있다."
    elif intent == "greeting":
        return "사용자는 인사를 건네고 있다."
    else:
        return "사용자는 다양한 질문을 하고 있다."
    
def fix_name(text):
    return text.replace("마나스의 토마토", "마나스톰이") \
               .replace("마나스토미", "마나스톰이") \
               .replace("마나스토마이", "마나스톰이") \
               .replace("마나스톰이너스", "마나스톰이")

def clean_answer(text: str) -> str:
    # 1️⃣ Assistant 이후 제거
    text = text.split("Assistant:")[0]

    # 2️⃣ 괄호 제거 (내용 포함)
    text = re.sub(r"\(.*?\)", "", text)

    # 3️⃣ 영어 제거 (선택)
    text = re.sub(r"[A-Za-z]+", "", text)

    # 4️⃣ 이상한 반복 줄이기 (마나스토마나스토 같은거)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)

    # 5️⃣ 공백 정리
    text = text.strip()

    return text
def force_fix_name(text):
    # 1️⃣ "나는 마나..." 패턴 교정
    text = re.sub(r"나는\s*마나[^\s!,.]*", "나는 마나스톰이다", text)

    # 2️⃣ "마나스톰이너스" 같은 변형 교정
    text = re.sub(r"마나[^\s!,.]*", "마나스톰", text)

    # 3️⃣ "나는 마나스톰이다"가 없을 때만 prefix 추가 (identity 상황에서만!)
    if "나는 마나스톰이다" not in text:
        text = "나는 마나스톰이다! " + text

    return text

    return text
def prefix_allowed_tokens_fn(batch_id, input_ids):
    text = tok.decode(input_ids, skip_special_tokens=True)

    banned_tokens = set()

    if "너" in text:
        # "너" 관련 토큰만 제외
        for word in ["너", "너는", "너가", "너를"]:
            ids = tok(word, add_special_tokens=False).input_ids
            banned_tokens.update(ids)

    return [i for i in range(tok.vocab_size) if i not in banned_tokens]

def run_inference(user_input: str):

    # 🔥 intent 분석
    intent = classify_intent(user_input)
    print("의도 @@@@ : ", intent)

    system_hint = make_system_hint(intent)
    #system_hint = make_system_hint("identity") #일단 자기소개 고정

    prompt = f"""### Instruction:
    너는 마나스톰이다.
    {system_hint}

    ### Input:
    {user_input}

    ### Response:
    """
    print("pronpt : ", prompt)
    print("프롬프트끝==================================")
    
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    answer = ""

    # 🔥 최대 3번 재시도
    for _ in range(3):
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=25,
                min_new_tokens=10,
                do_sample=True,
                temperature=0.4,
                top_p=0.7,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=tok.eos_token_id,
            )

        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        answer = tok.decode(gen_tokens, skip_special_tokens=True).strip()
        answer = clean_answer(answer)
        answer = keep_korean(answer)
        answer = fix_name(answer)
        if "다." in answer:
            answer = answer.split("다.")[0] + "다."
        
        if "!" in answer:
            parts = answer.split("!")
            answer = parts[0] + "!"
            if len(parts) > 1 and parts[1].strip():
                answer += " " + parts[1].strip() + "!"
                
        if not answer.endswith("!"):
            answer = answer.split("!")[0] + "!"
                            
        # 🔥 한글 체크
        if not contains_non_korean(answer) and len(answer) > 5:
            break

    return {
        "input": user_input,
        "prompt": prompt,
        "answer": answer,
        "mode": "QLoRA (4bit)" if isQLora else "LoRA (FP16)"
    }

# ============================================================
# ▶️ 직접 실행용
# ============================================================
'''
if __name__ == "__main__":
    test_input = "넌 누구냐"
    result = run_inference(test_input)
    print(force_fix_name(result["answer"]))
'''