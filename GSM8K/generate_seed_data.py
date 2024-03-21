import random
import os
import json

with open('grade-school-math/grade_school_math/data/train.jsonl', 'r') as f:
    data = f.readlines()
    
SUBSAMPLE_SPLIT = 0.01

subsampled_data = random.sample(data, int(SUBSAMPLE_SPLIT * len(data)))


formatted_data = []
for dp in subsampled_data:
    dp = eval(dp)
    formatted_data.append({
        "instruction": dp['question'],
        "output": dp['answer'],
        "input": "",
        "question_type": "gsm",
        "use_cot": True
    })
    
if not os.path.exists('data'):
    os.makedirs('data')

with open(f'data/gsm_{SUBSAMPLE_SPLIT}.json', 'w') as f:
    json.dump(formatted_data, f)