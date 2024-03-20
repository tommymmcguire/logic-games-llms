import csv
import json

def parse_base_eval(file_path: str):
    """
    Parse the unstructured output from the base 7B model evaluation.
    Determine if the model output etailment, non-entailment, unsure, or off topic.
    """
    parsed_data = {"entailment": [], "non-entailment": [], "unsure": [], "off-topic": []}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = data[1:]
    for row in data:
        if row:
            index = row[0]
            response = row[2]
            # Unsure
            if " entail" in response and "non-entail" in response:
                parsed_data["unsure"].append(index)
            # Non-entailment
            elif "non-entail" in response:
                parsed_data["non-entailment"].append(index)
            # Entailment
            elif "entail" in response:
                parsed_data["entailment"].append(index)
            # Off-topic
            else:
                parsed_data["off-topic"].append(index)
    # Write to JSON
    with open("../data/parsed_base_eval_responses.json", "w") as f:
        json.dump(parsed_data, f, indent=4)


if __name__ == "__main__":
    file_path = "../data/base_eval_responses.csv"
    parse_base_eval(file_path)
