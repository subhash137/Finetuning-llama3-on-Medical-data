
# Finetuning LLaMA-3 using Unsloth

## Description
This project aims to finetune the LLaMA-3 language model using the Unsloth library and a dataset of patient-doctor conversations. The goal is to improve the model's performance in understanding and generating medical dialogues.

## Dataset
The dataset used for finetuning is the `med_dialog` dataset from Hugging Face, which contains conversations between patients and doctors. The dataset can be accessed at the following link: https://huggingface.co/datasets/lighteval/med_dialog?row=0

Example from the dataset:
Patient: I am 57 years old man. I had a problem of looking at the places of lower sex organs of male and female and afraid and not able to talk to them freely. This was started 27 years back and even though I took various medicines advised by psychiatrists I am not cured of completely. At present I am using paxidep 12.5 mg and lonazep 0,25 mg. Can anyboxy advise me what to do?
Doctor: It has been so many yrs that you are facing a difficult problem.it is ocd dear or obsessive and compulsive disorder. Ssri helps but the dose you are taking is very less to be effective. Visit a psychiatrist, ask for increasing dose or change to fluoxetine.
## Model
The base model used for finetuning is the LLaMA-3 model, a large language model developed by Meta AI.

## What is Finetuning?
Finetuning is the process of taking a pre-trained language model and further training it on a specific task or domain-specific dataset. This process allows the model to adapt to the nuances and vocabulary of the target domain, resulting in better performance on that domain.

## About Unsloth and its Advantages over Hugging Face
Unsloth is a library for efficient and scalable finetuning of large language models. It offers several advantages over the popular Hugging Face library, including:

- Faster finetuning: Unsloth uses advanced techniques like gradient checkpointing and CPU offloading, which can significantly speed up the finetuning process.
- Efficient memory usage: Unsloth optimizes memory usage, enabling finetuning on larger models with limited GPU memory.
- Scalability: Unsloth supports distributed training across multiple GPUs and machines, allowing for faster and more efficient finetuning of large models.

## Finetuning Algorithm: Supervised Fine-tuning Trainer (SFT)
The algorithm used for finetuning in this project is the Supervised Fine-tuning Trainer (SFT). SFT is a crucial step in the Reinforcement Learning from Human Feedback (RLHF) process, which aims to align language models with human preferences.

Supervised fine-tuning involves training the language model on a labeled dataset, where the model learns to generate outputs that match the provided labels or target texts. In the case of this project, the model is trained on the `med_dialog` dataset, where the target outputs are the doctor's responses in the conversations.

## Unsloth Link
The Unsloth library can be found at the following GitHub repository: https://github.com/unslothai/unsloth

## Installation
To install the required dependencies, run the following commands:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install xformers transformers datasets torch trl

## Resources
The finetuning process requires significant computational resources. In this project, the finetuning is performed on a Google Colab instance with a T4 GPU (15GB VRAM). While this configuration is sufficient for finetuning, it may not provide optimal performance for faster inference.It needs more than 15gb vram for average inference.

For faster inference, it is recommended to use a system with more VRAM or to employ model quantization techniques. One such technique is to save the model in the GGUF (Grouped Gradient Unified Format) format, which can significantly reduce the model's memory footprint, allowing it to run efficiently on CPUs or lower-end GPUs.

To save the model in GGUF format, you can use the following code:

```python
model.save_pretrained_gguf("dir", tokenizer, quantization_method="q4_k_m")
