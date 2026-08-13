# 🤖 HR Assistant — Fine-Tuned LLM

A domain-specific **HR Assistant LLM** created by fine-tuning a pretrained language model using the Hugging Face ecosystem.

The goal of this project is to understand how a pretrained Large Language Model (LLM) can be adapted for HR-related conversations using **Supervised Fine-Tuning (SFT)**.

## 🚀 Project Overview

This project demonstrates the complete basic workflow of LLM fine-tuning:

```text
Pretrained LLM
      ↓
Training Dataset
      ↓
Data Preparation
      ↓
Tokenization
      ↓
Supervised Fine-Tuning
      ↓
Fine-Tuned Model
      ↓
Testing & Inference
      ↓
HR Assistant
```

## 🤗 Model

The trained model is available on Hugging Face:

**Model:** `bittu123456/hr_asstant_model`

**Hugging Face:**
https://huggingface.co/bittu123456/hr_asstant_model

The model is based on the Qwen2 family and is designed for conversational text generation.

## 🛠️ Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Hub
* Supervised Fine-Tuning (SFT)
* Large Language Models (LLMs)
* Natural Language Processing (NLP)

## 📁 Project Structure

```text
hr-assistant/
│
├── data/
│   └── dataset.json
│
├── training/
│   └── train.py
│
├── inference/
│   └── inference.py
│
├── requirements.txt
│
└── README.md
```

## 📊 Dataset

The model is trained using conversational HR-related examples.

A typical training example can look like:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the purpose of an HR department?"
    },
    {
      "role": "assistant",
      "content": "The HR department manages employee-related activities such as recruitment, onboarding, training, performance management and employee support."
    }
  ]
}
```

The training data is converted into a format that the language model can understand.

## 🧠 Fine-Tuning Process

### 1. Load the pretrained model

The pretrained model is loaded using Hugging Face Transformers.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name
)
```

### 2. Prepare the dataset

The HR conversation dataset is loaded and processed before training.

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="data/dataset.json"
)
```

### 3. Tokenization

The tokenizer converts text into tokens that the model can process.

```python
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length"
    )

tokenized_dataset = dataset.map(
    tokenize_function
)
```

### 4. Fine-Tune the Model

The model is trained using the prepared HR dataset.

The main idea is:

```text
Pretrained Knowledge
        +
HR Training Examples
        ↓
Fine-Tuned HR Assistant
```

Fine-tuning helps the model learn the style, patterns and type of responses required for the target task.

## 💻 Running the Model

Install the required packages:

```bash
pip install -r requirements.txt
```

Then run the inference script:

```bash
python inference/inference.py
```

## 🔥 Example Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "bittu123456/hr_asstant_model"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id
)

prompt = "What are the responsibilities of an HR manager?"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

outputs = model.generate(
    **inputs,
    max_new_tokens=150
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(response)
```

## 🧪 Example Questions

You can test the model with questions such as:

```text
What does an HR manager do?

How should a company handle employee onboarding?

What is the purpose of an HR interview?

How can HR improve employee engagement?

What is performance management?

What should be included in an employee policy?
```

## 📚 What I Learned

Through this project, I learned about:

* How pretrained LLMs work
* Preparing datasets for fine-tuning
* Tokenization
* Supervised Fine-Tuning (SFT)
* Hugging Face Transformers
* Model training
* Model evaluation
* Saving trained models
* Uploading models to Hugging Face Hub
* Running inference with a fine-tuned model

## ⚠️ Limitations

This model is an experimental project and should not be treated as a professional HR or legal advisor.

The model may sometimes:

* Generate incorrect information
* Produce incomplete answers
* Hallucinate information
* Give answers that require human verification

For production HR systems, additional techniques such as **RAG, evaluation pipelines, guardrails and human review** should be considered.

## 🔮 Future Improvements

Planned improvements include:

* [ ] Improve the training dataset
* [ ] Add more HR-specific conversations
* [ ] Experiment with LoRA
* [ ] Experiment with QLoRA
* [ ] Explore PEFT
* [ ] Add proper model evaluation
* [ ] Build an HR Assistant API
* [ ] Add RAG for company-specific HR documents
* [ ] Build a web interface
* [ ] Add safety and hallucination checks

## 🎯 Future Architecture

The next version can combine fine-tuning with RAG:

```text
                 User
                   ↓
            HR Assistant
                   ↓
          Fine-Tuned LLM
                   ↑
                   |
             RAG Pipeline
                   ↑
          HR Documents / PDFs
                   ↓
             Vector Database
```

Fine-tuning can teach the model **how to respond**, while RAG can provide **up-to-date and company-specific information**.

## 👨‍💻 Author

**Bittu Kumar**

AI/ML Engineer & Data Scientist

Interested in:

* Artificial Intelligence
* Machine Learning
* Large Language Models
* Generative AI
* RAG
* Agentic AI
* NLP

## ⭐ Acknowledgement

This project was built as a hands-on learning project to understand the practical process of fine-tuning Large Language Models.

If you find this project useful, consider giving the repository a ⭐.
