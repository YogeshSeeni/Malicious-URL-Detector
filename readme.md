# Bi-Directional Transformer Models for Phishing URL Detection

## Overview

This project evaluates the performance of various bi-directional transformer neural networks in classifying malicious phishing URLs. The study compares BERT, ALBERT, RoBERTa, and DistilBERT models using a dataset of over 96,000 URLs.

## Dataset

- PhishStorm dataset
- 96,018 URLs (48,009 benign, 48,009 malicious)
- 80% used for training, 20% for evaluation

## Models Evaluated

- BERT
- ALBERT
- RoBERTa
- DistilBERT

## Performance Metrics

- Accuracy
- F1-score
- Precision
- Recall
- Training Loss
- Validation Loss

## Key Results

- BERT achieved the highest performance with 98.59% accuracy and F1-score
- DistilBERT closely followed BERT in performance
- ALBERT showed slightly lower performance but offers a balance between efficiency and accuracy
- RoBERTa underperformed in this specific task

## Technologies Used

- Python
- Hugging Face Transformers library
- Pandas
- Scikit-learn

## Setup and Usage

```python
# Example code for model creation
from transformers import AutoModelForSequenceClassification

models = ['bert-base-uncased', 'albert-base-v2', 'roberta-base', 'distilbert-base-uncased']
CURRENT_MODEL = models[0]
model = AutoModelForSequenceClassification.from_pretrained(CURRENT_MODEL, num_labels=2)

# Training and evaluation
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./models',
    num_train_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_hg,
    eval_dataset=valid_hg,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
```

## Future Directions

- Tune hyperparameters for potentially improved performance
- Explore integration into real-world systems (e.g., browser extensions)
- Investigate hybrid models and novel transformer architectures

## Acknowledgements

This project was conducted as part of an AP Research paper. The PhishStorm dataset was used for training and evaluation.
