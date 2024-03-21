"""
Adapted from https://github.com/tatsu-lab/stanford_alpaca
"""

import concurrent.futures
import json
import os
import random
import re
import string
import time
from functools import partial
from multiprocessing import Pool

import fire
import numpy as np
import tqdm
from rouge_score import rouge_scorer

# Resolve relative import from parent directory
import sys
from pathlib import Path
utils_path = str(Path(__file__).resolve().parent.parent)
if utils_path not in sys.path:
    sys.path.append(utils_path)
    
import utils


def parallel_execution(f, n, inputs):
    # Add jittering
    def f_with_jitter(input_dict):
        time.sleep(random.uniform(0, 0.1))  # sleep for a random time between 0 and 0.1 seconds
        return f(input_dict)

    # Check if the length of inputs is equal to n
    if len(inputs) != n:
        raise ValueError("The length of the input list must be equal to the integer n")

    # Run the function calls in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the function f_with_jitter to the inputs and return the results as a list
        results = list(executor.map(f_with_jitter, inputs))

    return results

example = """The student was given the following question:

Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?

The answer key has this as the rationale and answer:

In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n#### 5

Please generate 1 similar question, along with the correct calculations and rationale."""


prompt_response = """Sure, I can help you with that! Here's a new math question based on the same scenario, along with the correct calculations and rationale:

1. Question: Betty is saving money for a new wallet which costs $150. Betty has already saved $30. Her parents decided to give her $20 for that purpose, and her grandparents gave her three times as much as her parents. How much more money does Betty need to buy the wallet?
Answer: In the beginning, Betty has saved $30. Betty's parents gave her $20. Betty's grandparents gave her 3 * $20 = $60. Total amount Betty has: $30 + $20 + $60 = $110. The cost of the wallet is $150. Amount more needed: $150 - $110 = $40.
#### 40"""

prompt_template = """The student was given the following question: 

{question}

The answer key has this as the rationale and answer:

{answer}

Please generate {num_aug} similar questions, along with the correct calculations and rationale."""

def encode_prompt(prompt_instructions, num_aug=1, current_keep=None):
    """Encode multiple prompt instructions into a single string."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt = open(f"{current_dir}/prompt_gsm.txt").read() + "\n"

    assert(len(prompt_instructions) == 1)
    if not current_keep:
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": example},
            {"role": "assistant", "content": prompt_response},
            {"role": "user", "content": prompt_template.format(question=prompt_instructions[0]["question"], answer=prompt_instructions[0]["answer"], incorrect_answer=prompt_instructions[0]["ai_answer"], num_aug=num_aug)},
        ]
    else:
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": example},
            {"role": "assistant", "content": prompt_response},
            {"role": "user", "content": prompt_template.format(question=prompt_instructions[0]["question"], answer=prompt_instructions[0]["answer"], incorrect_answer=prompt_instructions[0]["ai_answer"], num_aug=len(current_keep))},
        ]
        repeat_prompt = f"Here are {len(current_keep)} similar math questions based on the given scenario, along with the correct calculations and rationale:"
        for idx, task_dict in enumerate(random.sample(current_keep, len(current_keep))):
            (question, answer) = task_dict["question"], task_dict["answer"]
            repeat_prompt += f"\n{idx + 1}. Question: {question}\nAnswer:\n{answer}\n"
        messages.append({"role": "assistant", "content": repeat_prompt})
        messages.append({"role": "user", "content": f"Now generate {num_aug} more questions different from the ones you just generated, along with the correct calculations and rationale."})
    return messages


def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = response["message"]["content"]
    raw_instructions = re.split("[0-9]{1,2}\.\s+Question:", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"(Answer):", inst)
        
        if len(splitted_data) != 3:
            continue
        
        answers = re.split(f"(####)", splitted_data[2])
        if len(answers) != 3:
            continue
        question = splitted_data[0].strip()
        answer_digit = answers[2].split("\n", 1)[0].strip()
        answer = f"{answers[0].strip()}\n#### {answer_digit}"
        
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"question": question, "answer": answer})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_data(
    output_dir="",
    seed_tasks_path="",
    instructions_per_seed_task=4,
    generated_instructions_per_seed_task=4,
    model_name="gpt-3.5-turbo",
    num_prompt_instructions=1,
    request_batch_size=1,
    temperature=1.0,
    top_p=1.0,
    num_cpus=64,
    rouge_score_threshold=0.95,
    generation_workers=10,
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"question": t["question"], "answer": t["answer"], "ai_answer": t["ai_answer"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=instructions_per_seed_task * len(seed_instruction_data))
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["question"]for d in seed_instruction_data] + [
        d["question"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]
    num_requests = 0

    request_idx = len(machine_instruction_data) // instructions_per_seed_task
    print(f"Starting from instruction {request_idx} with 0 tasks kept for the current seed")


    def generate_task_for_idx(task_idx):
        prompt_instructions = seed_instruction_data[task_idx]
        kept_tasks_for_current_seed = 0  # Number of kept tasks for the current seed
        attempt = 0
        current_keep = []

        while kept_tasks_for_current_seed < instructions_per_seed_task:

            batch_inputs = []
            messages = encode_prompt([prompt_instructions], num_aug=min(generated_instructions_per_seed_task + attempt, 10), current_keep=current_keep if attempt % 2 == 1 else [])
            batch_inputs.append(messages)
            decoding_args = utils.OpenAIDecodingArguments(
                temperature=temperature,
                n=1,
                max_tokens=2149,
                top_p=top_p,
            )
            request_start = time.time()
            def f(message):
                result = utils.openai_completion(
                    prompts=message,
                    model_name=model_name,
                    batch_size=request_batch_size,
                    decoding_args=decoding_args,
                    logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
                )
                return result

            all_results = parallel_execution(f, request_batch_size, batch_inputs)
            nonlocal num_requests
            num_requests += 1
            request_duration = time.time() - request_start

            process_start = time.time()
            instruction_data = []
            for results in all_results:
                for result in results:
                    new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
                    instruction_data += new_instructions

            total = len(instruction_data)
            keep = 0
            for instruction_data_entry in instruction_data:
                # computing similarity with the pre-tokenzied instructions
                new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["question"])
                with Pool(num_cpus) as p:
                    rouge_scores = p.map(
                        partial(rouge_scorer._score_lcs, new_instruction_tokens),
                        all_instruction_tokens,
                    )
                rouge_scores = [score.fmeasure for score in rouge_scores if score is not None]
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                }
                if max(rouge_scores) > rouge_score_threshold:
                    continue
                else:
                    keep += 1
                    kept_tasks_for_current_seed += 1  # Increase the number of kept tasks for this seed
                    current_keep.append(instruction_data_entry)

                instruction_data_entry["most_similar_instructions"] = most_similar_instructions
                instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
                instruction_data_entry["seed"] = prompt_instructions
                machine_instruction_data.append(instruction_data_entry)
                all_instructions.append(instruction_data_entry["question"])
                all_instruction_tokens.append(new_instruction_tokens)
                progress_bar.update(1)

                if kept_tasks_for_current_seed >= instructions_per_seed_task:
                    break
            process_duration = time.time() - process_start
            print(f"Instruction {task_idx} Request {num_requests} Attempt {attempt} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
            print(f"Generated {total} instructions, kept {keep} instructions")
            utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))

            attempt += 1
    

    with concurrent.futures.ThreadPoolExecutor(max_workers=generation_workers) as executor:
        print(f"Submitting tasks for request idx {request_idx} to { len(seed_instruction_data)}")

        futures = [executor.submit(generate_task_for_idx, idx) for idx in range(request_idx, len(seed_instruction_data))]

        for future in concurrent.futures.as_completed(futures):
            future.result()
    

def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
