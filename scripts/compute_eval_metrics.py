import json


def compute_eval_metrics(target_path: str, model_output_path: str):
    """
    Compute the evaluation metrics for the model output.

    Args:
    target_path (str): The path to the target data in json format.
    model_output_path (str): The path to the model output in json format.

    Returns:
    dict: A dictionary containing the evaluation metrics.
    """
    with open(target_path, "r") as f:
        target_data = json.load(f)
    with open(model_output_path, "r") as f:
        model_output = json.load(f)

    # Initialize the evaluation metrics
    eval_metrics = {"total_accuracy": 0, "entailment_accuracy": 0, "non-entailment_accuracy": 0, "unsure_total": 0, "off-topic_total": 0, 
                    "total_entailment_in_target": 0, "total_non-entailment_in_target": 0, "total_entailment_in_output": 0, "total_non-entailment_in_output": 0,
                    "output_entail_when_non-entail": 0, "output_non-entail_when_entail": 0, "output_unsure_when_entail": 0, "output_unsure_when_non-entail": 0,
                    "output_off-topic_when_entail": 0, "output_off-topic_when_non-entail": 0}

    # Find the ground truth
    ground_truth = {"entailment": [], "non-entailment": []}
    for i, d in enumerate(target_data):
        if i < 500:
            label = d["output"]
            if "not entailment" in label:
                ground_truth["non-entailment"].append(i)
            elif "entailment" in label:
                ground_truth["entailment"].append(i)

    # Find the eval metrics
    for key in model_output:
        if key == "entailment":
            # Find entailment accuracy
            correct = len(set([int(i) for i in ground_truth[key]]).intersection(set([int(i) for i in model_output[key]])))
            total_in_target = len(ground_truth[key])
            total_in_output = len(model_output[key])
            eval_metrics["total_entailment_in_target"] = total_in_target
            eval_metrics["total_entailment_in_output"] = total_in_output
            eval_metrics["entailment_accuracy"] = correct/total_in_target
            eval_metrics["total_accuracy"] = (correct + eval_metrics["non-entailment_accuracy"])/500
            # Find the output_entail_when_non-entail
            total_output_entail_when_non_entail = set([int(i) for i in model_output[key]]).intersection(set(ground_truth["non-entailment"]))
            eval_metrics["output_entail_when_non-entail"] = len(total_output_entail_when_non_entail)
            
        if key == "non-entailment":
            # Find non-entailment accuracy
            correct = len(set([int(i) for i in ground_truth[key]]).intersection(set([int(i) for i in model_output[key]])))
            total_in_target = len(ground_truth[key])
            total_in_output = len(model_output[key])
            eval_metrics["total_non-entailment_in_target"] = total_in_target
            eval_metrics["total_non-entailment_in_output"] = total_in_output
            eval_metrics["non-entailment_accuracy"] = correct/total_in_target
            eval_metrics["total_accuracy"] = (correct + eval_metrics["entailment_accuracy"])/500
            # Find the output_non-entail_when_entail
            total_output_non_entail_when_entail = set([int(i) for i in model_output[key]]).intersection(set(ground_truth["entailment"]))
            eval_metrics["output_non-entail_when_entail"] = len(total_output_non_entail_when_entail)

        if key == "unsure":
            eval_metrics["unsure_total"] = len(model_output[key])
            # Find the output_unsure_when_entail
            total_output_unsure_when_entail = set([int(i) for i in model_output[key]]).intersection(set(ground_truth["entailment"]))
            eval_metrics["output_unsure_when_entail"] = len(total_output_unsure_when_entail)
            # Find the output_unsure_when_non-entail
            total_output_unsure_when_non_entail = set([int(i) for i in model_output[key]]).intersection(set(ground_truth["non-entailment"]))
            eval_metrics["output_unsure_when_non-entail"] = len(total_output_unsure_when_non_entail)
            
        if key == "off-topic":
            eval_metrics["off-topic_total"] = len(model_output[key])
            # Find the output_off-topic_when_entail
            total_output_off_topic_when_entail = set([int(i) for i in model_output[key]]).intersection(set(ground_truth["entailment"]))
            eval_metrics["output_off-topic_when_entail"] = len(total_output_off_topic_when_entail)
            # Find the output_off-topic_when_non-entail
            total_output_off_topic_when_non_entail = set([int(i) for i in model_output[key]]).intersection(set(ground_truth["non-entailment"]))
            eval_metrics["output_off-topic_when_non-entail"] = len(total_output_off_topic_when_non_entail)

    return eval_metrics


if __name__ == "__main__":
    target_path = "../data/sft-data/test.json"
    model_output_path = "../data/parsed_base_eval_responses.json"
    eval_metrics = compute_eval_metrics(target_path, model_output_path)
    # Write to JSON
    with open("../data/base_eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)
    for key, value in eval_metrics.items():
        print(f"{key}: {value}")
    
