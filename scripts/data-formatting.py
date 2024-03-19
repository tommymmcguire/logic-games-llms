import json

files = ['data/downloaded-data/train.jsonl', 'data/downloaded-data/validation.jsonl', 'data/downloaded-data/test.jsonl']

for file_path in files:
    litgpt_format_data = []

    # Open the .jsonl file and process each line
    with open(file_path, 'r') as file:
        for line in file:
            example = json.loads(line)  # Parse each line as JSON
            new_example = {
                "instruction": "Based off this context, output whether the question implies entailment or non-entailment",
                "input": f"Conetext: {example['puzzle_text']}, Question: {example['question']}",
                "output": example["answer"]+'\n',
            }
            litgpt_format_data.append(new_example)

    # Derive the new filename for the formatted data
    name = file_path.split('/')[-1].split('.')[0]
    output_file_path = f'data/formatted-data/{name}_data_formatted.json'

    # Dump the converted data to a JSON file
    with open(output_file_path, 'w') as json_file:
        json.dump(litgpt_format_data, json_file, indent=4)
