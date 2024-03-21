import argparse
import json
import re

def clean_string(s):
    return ''.join(c for c in s if c.isdigit() or c == '.')

def compare_strings(s1, s2):
    try:
        num1 = float(clean_string(s1))
        num2 = float(clean_string(s2))
        return num1 == num2
    except:
        return s1.lower().replace(",", "") == s2.lower().replace(",", "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process json files')
    parser.add_argument('--seed_data_path', type=str)
    parser.add_argument('--input_data_path', type=str)
    parser.add_argument('--output_data_path', type=str)
    parser.add_argument('--no_filter', action='store_true')

    args = parser.parse_args()

    filtered_data = []
    answers = {dp["instruction"]: dp["output"] for dp in json.load(open(args.seed_data_path))}


    with open(args.input_data_path) as f:
        input = f.readlines()
        

    for dp in input:
        result = json.loads(dp)
        if result["question"] in answers:
            gold_label = answers[result["question"]]
            output = result["text"]

            answer = re.search(r'#### (.*)', gold_label, re.DOTALL)[1]
            ai_answer = re.search(r'Answer:(.*)', result["text"], re.DOTALL)[1].strip()
            
            try:
                response_answer = re.search(r"#+ (.*)", result["text"]).group(1).strip()
            except:
                response_answer = "WRONG"

            if not compare_strings(answer, response_answer) or args.no_filter:
                filtered_data.append({"question": result["question"], "answer": gold_label, "ai_answer": ai_answer})


    with open(args.output_data_path, "w") as f:
        for dp in filtered_data:
            f.write(json.dumps(dp) + "\n")

