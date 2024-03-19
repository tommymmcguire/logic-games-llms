import random
import json

def sample_data(file_path, output_path, num_samples):
    '''
    Sample a number of examples from a json file and save to new file
    Arg: file_path - path to the file
         output_path - path to the output file
         num_samples - number of samples to draw
    '''
    with open(file_path, 'r') as f:
        data = json.load(f)
    samples = random.sample(data, num_samples)
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=4)
    return samples

if __name__ == '__main__':
    sample_data('data/formatted-data/train_data_formatted.json', 'data/sft-data/train.json', 5000)
    sample_data('data/formatted-data/validation_data_formatted.json', 'data/sft-data/val.json', 750)
    sample_data('data/formatted-data/test_data_formatted.json', 'data/sft-data/test.json', 750)
    print('Sample data saved to data/sft-data/...')