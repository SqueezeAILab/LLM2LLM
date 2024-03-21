import os
import json
import tqdm
import copy
import shortuuid
from typing import List, Union, Any
from pathlib import Path
from itertools import islice
import argparse
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

def get_json(file):
    with open(file, "r") as f:
        return json.load(f)
def write_json_list(results: List[Any], filename: Union[Path, str]):
    with open(filename, "w") as f:
        for item in results:
            json.dump(item, f)
            f.write("\n")


template = """Solve the below question step by step.

{question}

Answer:"""


def do_generate(model, tokenizer, questions, question_ids, max_new_tokens=100):
    question_texts = [q["question"] for q in questions]
    formatted_question_texts = [template.format(question=q_t) for q_t in question_texts]

    tokens = tokenizer(formatted_question_texts, return_tensors='pt', padding=True, truncation=True)
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask
    full_completions = model.generate(
        inputs=input_ids.cuda(),
        attention_mask=attention_mask.cuda(),
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        max_length=input_ids.shape[1] + max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    end_texts = [tokenizer.decode(fc, skip_special_tokens=True) for fc in full_completions]
    return end_texts

def enumerate_n_items(data, n):
    it = iter(data)
    index = 0
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield index, chunk
        index += n

# Generate answers and save them to a properly formatted answers jsonl
answer_template = json.loads('{"question_id": 0, "text": "", "answer_id": "", "model_id": "", "metadata": {}}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process json files')
    parser.add_argument('--mode', type=str, choices=['test', 'train'])
    parser.add_argument('--model', type=str)
    parser.add_argument('--file', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--output_path', type=str)
    question_type = "grade_school_math"


    path_to_model = parser.parse_args().model
    path_to_tokenizer = path_to_model

    model_name = os.path.basename(path_to_model)

    args = parser.parse_args()

    EVAL_BATCH_SIZE = args.batch_size

    # Load pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(path_to_tokenizer, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(path_to_model, device_map="auto", torch_dtype=torch.bfloat16)


    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    for i in range(args.iterations):

        # Get current python script file's directory
        current_script_path = os.path.dirname(os.path.realpath(__file__))

        if args.file:
            questions_path = args.file
            output_path = fr"{args.output_path}/file_{i}.jsonl"
            questions = [{"question": dp["instruction"], "answer": dp["output"]} for dp in json.load(open(questions_path))]
        elif args.mode == "test":
            questions_path = f"{current_script_path}/grade-school-math/grade_school_math/data/test.jsonl"
            output_path = fr"{args.output_path}/test_{i}.jsonl"
            questions = get_json_list(questions_path)
        elif args.mode == "train":
            questions_path = f"{current_script_path}/grade-school-math/grade_school_math/data/train.jsonl"
            output_path = fr"{args.output_path}/train_{i}.jsonl"
            questions = get_json_list(questions_path)
        else:
            raise Exception("Invalid mode")

        with open(output_path, "w") as f:       
            for initial_idx, qs_batch in enumerate_n_items(tqdm.tqdm(questions), EVAL_BATCH_SIZE):
                try:
                    answers = do_generate(model, tokenizer, qs_batch, [initial_idx + offset for offset in range(len(qs_batch))])
                except:
                    import traceback
                    traceback.print_exc()
                    print(f"Failed to find answer for question, please check manually")
                for offset, answer in enumerate(answers):
                    answer_dict = copy.copy(answer_template)
                    answer_dict["question_id"] = initial_idx + offset
                    answer_dict["question"] = qs_batch[offset]["question"]
                    answer_dict["answer"] = qs_batch[offset]["answer"]
                    answer_dict["text"] = answer
                    answer_dict["answer_id"] = shortuuid.uuid()
                    answer_dict["model_id"] = model_name
                    json.dump(answer_dict, f)
                    f.write("\n")
                    f.flush()
