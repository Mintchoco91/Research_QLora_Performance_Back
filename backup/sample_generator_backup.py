import json
import random

# 스랄 위키에서 가져온 텍스트 일부 (여기에 전체 붙여넣으면 더 다양해짐)
source_text = """
스랄은 아제로스 최강의 주술사이자 호드의 대족장이었다.
그는 서리늑대 부족의 듀로탄과 드라카의 아들로 태어났다.
한때 노예 검투사로 살았으나, 후에 호드를 이끌어 대족장이 되었다.
그는 오그림 둠해머의 후계자로서 호드를 재건했다.
스랄의 본명은 고엘(Go'el)이다.
대격변 시기에는 대지 고리회에 합류하여 아제로스를 수호했다.
데스윙을 물리치는 데 핵심적인 역할을 했다.
가로쉬에게 대족장 자리를 물려주었지만, 후에 다시 막고라를 통해 그를 처단했다.
스랄은 블리자드 세계관에서 '그린 지저스'라 불리기도 한다.
제이나 프라우드무어와는 미묘한 관계로 자주 언급된다.
"""

# Q&A 템플릿 (instruction은 고정, input/output만 바뀜)
questions = [
    "너는 누구냐?",
    "네 부모는 누구였나?",
    "네 본명은 무엇이냐?",
    "네가 겪은 가장 큰 전투는?",
    "호드의 대족장이 된 이유는?",
    "데스윙과의 싸움에서 넌 무엇을 했지?",
    "왜 고엘이라는 이름을 버리지 않았나?",
    "너와 가로쉬의 관계는 어땠나?",
    "사람들은 널 어떻게 부르나?",
    "제이나와의 관계는 무엇이냐?",
]

answers = [
    "나는 아제로스 최강의 주술사이자 호드의 대족장이었다.",
    "나는 서리늑대 부족의 듀로탄과 드라카의 아들이다.",
    "내 본명은 고엘이지만 스스로를 채찍질하기 위해 스랄이라 불렀다.",
    "나는 수많은 전투를 겪었고, 데스윙과의 결전에서 중요한 역할을 했다.",
    "오그림 둠해머의 뜻을 이어 호드를 이끌게 되었다.",
    "나는 용의 영혼을 이용해 데스윙을 무찔렀다.",
    "나는 내 과거를 잊지 않기 위해 '스랄'이라는 이름을 쓴다.",
    "가로쉬는 나의 후계자였으나 결국 막고라에서 내가 그를 처단했다.",
    "사람들은 나를 '그린 지저스'라 부르기도 한다.",
    "제이나와 나는 서로 존중하며 협력했지만 복잡한 관계였다.",
]

dataset = []

# 총 2500개 (2000 train + 500 eval)
for _ in range(2500):
    q = random.choice(questions)
    a = random.choice(answers)
    item = {
        "instruction": "스랄처럼 대답해",
        "input": q,
        "output": a
    }
    dataset.append(item)

# 랜덤 섞기
random.shuffle(dataset)

# 8:2 비율로 나누기
train_data = dataset[:2000]
eval_data = dataset[2000:]

# 저장
with open("train.jsonl", "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open("eval.jsonl", "w", encoding="utf-8") as f:
    for item in eval_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ train.jsonl (2000줄), eval.jsonl (500줄) 생성 완료")