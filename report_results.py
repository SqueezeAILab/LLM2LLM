import json
import re
import os
import argparse

def load_seed_data(seed_data_path):
    with open(seed_data_path, 'r') as file:
        if seed_data_path.endswith("json"):
            return json.load(file)
        elif seed_data_path.endswith("jsonl"):
            return [json.loads(l) for l in file.readlines()]
        else:
            raise ValueError("invalid seed data")

def find_matching_seed(seed_data, instruction):
    for item in seed_data:
        if 'instruction' in item and item['instruction'] == instruction:
            return item['output'] # casehold
        if 'question' in item and item['question'] == instruction:
            return re.search(r'#### (.*)', item['answer'], re.DOTALL)[1]
    return None

def clean_string(s):
    return ''.join(c for c in s if c.isdigit() or c == '.')

def compare_strings(s1, s2):
    try:
        num1 = float(clean_string(s1))
        num2 = float(clean_string(s2))
        return num1 == num2
    except:
        return s1.lower().replace(",", "") == s2.lower().replace(",", "")

def compare_results(seed_data, eval_folder, results_file_name):
    print("Full Path,Correct Count,Total Matched,Accuracy")
    for root, dirs, files in os.walk(eval_folder):
        files.sort()
        dirs.sort()
        for file in files:
            if file == results_file_name:
                full_path = os.path.join(root,file)
                with open(full_path, 'r') as file:
                    eval_data = [json.loads(line) for line in file]

                correct_count = 0
                total_matched = 0

                for eval_item in eval_data:
                    matching_output = find_matching_seed(seed_data, eval_item['question'])

                    casehold_output_candidate = re.search(r"Answer:(.*)", eval_item["text"])
                    gsm_output_candidate = re.search(r"#+ (.*)", eval_item["text"])

                    if casehold_output_candidate:
                        casehold_output_candidate = casehold_output_candidate.group(1).strip()
                    if gsm_output_candidate:
                        gsm_output_candidate = gsm_output_candidate.group(1).strip()
                    
                    if matching_output == casehold_output_candidate or (matching_output is not None and gsm_output_candidate is not None and compare_strings(matching_output, gsm_output_candidate)):
                        correct_count += 1
                    if matching_output is not None:
                        total_matched += 1

                if total_matched > 0:
                    accuracy = (correct_count / total_matched) * 100
                    print(f"{full_path},{correct_count},{total_matched},{accuracy:.2f}")
                else:
                    print(f"{full_path}, No matches found")

def main():
    parser = argparse.ArgumentParser(description='Compare output and answer fields in JSON and JSONL files.')
    parser.add_argument('seed_data_json', type=str, help='Path to the seed data JSON file')
    parser.add_argument('eval_results_folder', type=str, help='Path to the evaluation results folder')
    parser.add_argument('--results_file_name', type=str, default="train_0.jsonl", help='Name of the results file (default: train_0.jsonl)')
    
    args = parser.parse_args()

    seed_data = load_seed_data(args.seed_data_json)
    compare_results(seed_data, args.eval_results_folder, args.results_file_name)

if __name__ == "__main__":
    main()
