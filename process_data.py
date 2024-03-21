import json
import argparse
import random

def combine_files(file1: str, file2: str, output_file: str, seed: int):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        list1 = json.load(f1)
        list2 = json.load(f2)

    combined = list1 + list2
    random.seed(seed)
    random.shuffle(combined)

    with open(output_file, 'w') as of:
        json.dump(combined, of, indent=4)

def convert_file(file: str, output_file: str, use_cot: bool, question_type: str):
    with open(file, 'r') as f:
        data = json.load(f)

    converted = []
    for obj in data:
        converted.append({
            "instruction": obj["question"],
            "output": obj["answer"],
            "input": "",
            "question_type": question_type,
            "use_cot": use_cot
        })

    with open(output_file, 'w') as of:
        json.dump(converted, of, indent=4)

    print(f'Processed {len(converted)} objects')

def reduce_file(file: str, output_file: str, remaining_file: str, percent: float):
    with open(file, 'r') as f:
        data = json.load(f)

    cutoff = int(len(data) * (percent / 100))

    # Split the data based on the calculated cutoff
    included_data = data[:cutoff]
    excluded_data = data[cutoff:]

    # Save the included data
    with open(output_file, 'w') as of:
        json.dump(included_data, of, indent=4)

    # Save the excluded data
    with open(remaining_file, 'w') as rf:
        json.dump(excluded_data, rf, indent=4)

    print(f'Saved {len(included_data)} objects to {output_file}')
    print(f'Saved {len(excluded_data)} objects to {remaining_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process json files')
    parser.add_argument('mode', type=str, choices=['combine', 'convert', 'reduce'])
    parser.add_argument('files', type=str, nargs='+')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--remaining', type=str, help='Filename to save the remaining data (used in reduce mode)')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--percent', type=float, help='Percentage of data to keep (used in reduce mode)', default=100)
    parser.add_argument('--use_cot', type=bool, default=True, help='Whether to use COT')
    parser.add_argument('--question_type', type=str, default='gsm')

    args = parser.parse_args()

    if args.mode == 'combine':
        if len(args.files) != 2:
            raise Exception('Combine mode requires exactly two input files')
        combine_files(args.files[0], args.files[1], args.output, args.seed)
    elif args.mode == 'convert':
        if len(args.files) != 1:
            raise Exception('Convert mode requires exactly one input file')
        convert_file(args.files[0], args.output, args.use_cot, args.question_type)
    elif args.mode == 'reduce':
        if len(args.files) != 1:
            raise Exception('Reduce mode requires exactly one input file')
        if not (0 <= args.percent <= 100):
            raise Exception('Percentage should be between 0 and 100')
        if args.remaining is None:
            raise Exception('The --remaining argument is required for the reduce mode')
        reduce_file(args.files[0], args.output, args.remaining, args.percent)

