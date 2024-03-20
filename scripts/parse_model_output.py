import csv
import json

def parse_model_output_csv(file_path: str, output_path: str):
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
            response = row[2].lower()
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
    with open(output_path, "w") as f:
        json.dump(parsed_data, f, indent=4)


if __name__ == "__main__":
    file_path = "../data/results/gpt3turbo-predictions.csv"
    output_path = "../data/results/parsed_gpt_eval_responses.json"
    parse_model_output_csv(file_path, output_path)
