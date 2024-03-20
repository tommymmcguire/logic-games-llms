from openai import OpenAI
import json
import os
import csv
from dotenv import load_dotenv


def load_test_data(test_data_path):
    '''
    Load test data from a json file
    Arg: test_data_path: path to the test data file
    Ret: test data as a list of dictionaries
    '''
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    return test_data

def query_gpt(prompt, model="gpt-3.5-turbo", temperature=1, max_tokens=300):
    """
    Call the OpenAI API to generate a response based on the prompt.

    Args:
        prompt (str): Prompt for the OpenAI API

    Returns:
        response (str): Response from the OpenAI API
    """
    # OpenAI Client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # OpenAI Completion API
    response = client.chat.completions.create(model=model,
                                            messages=[{"role": "user", "content": prompt}],
                                            temperature=temperature,
                                            max_tokens=max_tokens,
                                            frequency_penalty=0,
                                            presence_penalty=0
                                        )

    return response.choices[0].message.content.strip()

def generate_responses(test_data, output_csv, model="gpt-3.5-turbo"):
    '''
    Generate responses from the OpenAI API
    Args: test_data - list of dictionaries to run inferences
          model - model to use for generation
          output_file - path to save the output
    '''
    for i, d in enumerate(test_data):
        prompt = f"{d['input']}, {d['instruction']}"
        response = query_gpt(prompt, model)

        with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Check if file is empty to write headers
            if csvfile.tell() == 0:
                csvwriter.writerow(['Index', 'Prompt', 'Output'])
            csvwriter.writerow([i, prompt, response])
        
        if i % 10 == 0:
            print(f"Generated {i}/{len(test_data)} responses")
    
    print(f"Responses saved to {output_csv}")

if __name__ == '__main__':
    load_dotenv()
    test_data = load_test_data('data/sft-data/test.json')
    generate_responses(test_data, 'data/results/gpt3turbo-predictions.csv')
    print('Evaluation complete')