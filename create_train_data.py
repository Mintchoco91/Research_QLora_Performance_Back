from datasets import load_dataset

# 1. IMDB 데이터셋 불러오기
dataset = load_dataset("stanfordnlp/imdb")

# 2. train 데이터를 다시 쪼개서 (60:20 비율)
split_dataset = dataset["train"].train_test_split(test_size=0.25, seed=42)  
# => train 75% / valid 25%
# 75% 중에서 실제 비율은 (전체의 0.75 = 60%), (전체의 0.25 = 20%)

train_dataset = split_dataset["train"]   # 학습 데이터 - 전체의 60%
val_dataset   = split_dataset["test"]    # 검증 데이터 - 전체의 20%
test_dataset  = dataset["test"]          # 원래 제공된 test (전체의 20%)
