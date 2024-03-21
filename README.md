# Fine-tuning Mistral7B for Logical Entailment Detection

This project focuses on fine-tuning the Mistral7B-Instruct model using the LoRA optimization technique to enhance its ability to determine logical entailment from given contexts and questions.

## Project Overview

The goal of this project is to enhance the logical reasoning capabilities of the Mistral7B model, making it more adept at understanding and evaluating contexts that require a logical entailment determination. By leveraging the LoRA optimization technique, we aim to efficiently fine-tune the model on the specialized dataset with limited computational resources. 

### Data

The dataset used for fine-tuning is sourced from [huggingface - tasksource/puzzte](https://huggingface.co/datasets/tasksource/puzzte), which is specifically curated to challenge models with tasks requiring advanced logical reasoning. The dataset comprises various contexts and questions, where the objective is to determine whether the given context logically entails the question posed. We used 5000 samples from the dataset to form labeled pairs for use in supervised fine-tuning formatted in `data/sft-data`.

### Fine-tuning Methodology

We employed the LoRA (Low-Rank Adaptation) technique for fine-tuning the Mistral7B-Instruct model using the [litgpt](https://github.com/Lightning-AI/litgpt) package. LoRA allows for parameter efficient fine-tuning of large language models by modifying only a small subset of the model's parameters through low-ranked matricies. This method significantly reduces the computational overhead typically associated with training large models allowing us to use a single GPU for only a few hours.

### Evluation

In order to evaluate our results, we generated outputs from OpenAI GPT3.5-turbo, Mistral7B-Instruct, and our finetuned Mistral7B. We then compared the accuracy of each of these models on 500 test examples. We found that the finetuned model performed the best as well as outputted resposes much more aligned to the data set in formatting. Results can be found in `data/results`

| Model                           | Overall Accuracy | “Entailment” Accuracy | “Not Entailment” Accuracy |
|---------------------------------|------------------|-----------------------|---------------------------|
| GPT-3.5-turbo                   | 47.4%            | 61.4%                 | 40.1%                     |
| Mistral-7B-Instruct-v0.2        | 43.8%            | 26.3%                 | 51.4%                     |
| Mistral-7B-Finetuned            | 63.8%            | 77.8%                 | 56.5%                     |


### Resources

- **LitGPT:** Our fine-tuning process utilizes [lightningAI/litgpt](https://github.com/Lightning-AI/litgpt), a robust and flexible framework for working with large language models in PyTorch. This framework simplifies the training and deployment process, enabling us to implement LoRA finetuning.
- **Google Colab:** All experiments and fine-tuning processes were conducted on [Google Colab](https://colab.google/) using thier V100 and A100 GPUs.

## Repository Structure

The repository is organized into three main directories:

- `data/`: Contains the dataset used for fine-tuning and test, along results from training and evlaution.
- `scripts/`: Includes various Python scripts used for model evaluation, data preperation, and utility functions.
- `notebooks/`: Jupyter notebooks built with colab demonstrating the fine-tuning process and model evaluation.

## Getting Started

To replicate our fine-tuning process or to experiment with the fine-tuned model explore the Jupyter notebooks in the `notebooks/` directory for examples and walkthroughs of the fine-tuning and evaluation process using google colab.

Our fine-tuned model weights can be found here: [Mistral7B-finetuned](https://drive.google.com/file/d/1IkdLuO8Vkq8eT70-OXDVPBXBb68ibdC_/view?usp=drive_link)

## Acknowledgments

This project was made possible through the use of open-source software and publicly available datasets. We extend our gratitude to [MistralAI](https://mistral.ai/), [lightningAI/litgpt](https://github.com/Lightning-AI/litgpt), and [HuggingFace](https://huggingface.co/datasets).

---