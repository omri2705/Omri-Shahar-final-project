# Omri-Shahar-final-project
# 🏋️ Personalized Gym Plan Generator Using Fine-Tuned GPT-2

This project demonstrates how to train, fine-tune, and evaluate a GPT-2-based language model to generate personalized gym workout plans based on user input (e.g., age, BMI, goals). We compare multiple training approaches: full fine-tuning, LoRA (Low-Rank Adaptation), and different dataset sizes and epochs.

## 🚀 Project Summary

Our model generates weekly workout plans given a user's profile prompt. We trained different versions of GPT-2 on datasets of sizes ranging from 10 to 14,000 examples using:
- 1 epoch
- 3 epochs
- LoRA (parameter-efficient training)

We evaluate the generated outputs using:
- **BLEU**: N-gram precision-based score
- **ROUGE-L**: Longest common subsequence-based recall score
- **Semantic Similarity**: Using sentence embeddings from `all-MiniLM-L6-v2`

## 📂 Contents

│
├── colab/
│ └── evaluation_demo.ipynb ← Runs trained models on unseen examples + computes scores
│
├── kaggle/
│ └── training_pipeline.ipynb ← Code for training GPT-2 using full fine-tuning and LoRA
│
├── datasets/
│ └── eval_dataset.jsonl ← Evaluation dataset (100 examples not seen during training)
│
└── README.md


## 🧪 Evaluation Metrics

| Metric             | Description |
|--------------------|-------------|
| **BLEU**           | Measures n-gram overlap precision between output and reference |
| **ROUGE-L**        | Measures longest common subsequence (recall) |
| **Semantic Similarity** | Cosine similarity between model and reference embeddings |

All models are evaluated on the same 100 prompts and compared quantitatively.

## 📊 Demonstration (Colab)

The Colab notebook:
- Randomly selects one evaluation example
- Generates output using 5 different models:
  - Base GPT-2
  - 1 epoch (100 samples)
  - 3 epochs (100 samples)
  - LoRA (100 samples)
  - 3 epochs (14k samples)
- Computes and displays evaluation metrics + explanations

## 📚 Tools & Libraries

- `transformers` (Hugging Face)
- `evaluate` (for BLEU and ROUGE)
- `sentence-transformers`
- `torch`
- `jsonlines`
- Google Colab / Kaggle for execution

## 💡 Authors

- Omri & Shahar – Final Project, Electrical Engineering, 2025

---

