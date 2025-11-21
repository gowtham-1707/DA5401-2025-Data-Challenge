# LLM Evaluation Score Prediction using Sentence Embedding Regression + Pseudo-Labeling
## Overview
This project addresses the DA5401 Data Challenge, where the goal is to predict LLM evaluation scores (0–10) for multilingual prompt–response pairs using provided training and test data.
### The challenge includes:
Multilingual prompts (Tamil, Hindi, Bengali, Assamese, English, Sindhi, Bodo)
Score distribution heavily skewed toward high values
Missing system prompts in many instances
No direct metric-definition text (only embeddings provided)
Need to avoid model overfitting and leverage pseudo-labeling carefully

### This repository contains a PyTorch-based regression approach using:
SentenceTransformer "all-MiniLM-L6-v2"
Neural regression architecture (RegNet)
Training with data augmentation using pseudo-labels
Three-seed model ensembling
Output smoothing and controlled randomness to reduce prediction collapse

### Repository Structure
.
├── final_code.py        # Main training + inference script (uploaded)  
├── README.md            
├── sample_submission.csv (needed for pseudo labels)
├── train_data.json
└── test_data.json

### Main solution implementation:
final_code.py 
final_code
 Methodology Summary
1. Text Extraction

A custom extract_text() function was built to robustly pull text from multiple possible keys:
metric_name

Prompt variants: prompt, input, query, question

System instructions: system_prompt, instruction, system

Expected answer: expected_response, response, answer

Each record is converted to a single combined text string.

2. Pseudo-Label Augmentation

To expand the training data, the sample submission is used:

Only select high-confidence pseudo labels
Defined as:

score ≤ 2  OR  score ≥ 8


These test samples are appended to training data.

This helps stabilize the model around extreme scores.

3. Embedding Extraction

All combined texts are encoded using:

sentence-transformers/all-MiniLM-L6-v2


Advantages:

Lightweight (384-dim)

Fast inference

Good multilingual alignment

4. Regression Model (RegNet)

A custom PyTorch regressor:

Input → FC(768) → BN → ReLU → Dropout
      → FC(384) → BN → ReLU → Dropout
      → FC(128) → ReLU
      → FC → Output Score


Training details:

AdamW optimizer

BatchNorm + Dropout for generalization

Early stopping (patience 8)

MSE loss

Batch size = 96

Epochs = 60 max

5. Multi-Seed Ensembling

The model is trained with seeds:

[17, 88, 201]

Each model predicts test embeddings; outputs are averaged:

ensemble_pred = mean(predictions)

This reduces variance and smooths predictions significantly.

6. Post-Processing

To avoid predictions collapsing into narrow ranges:

Add Gaussian noise near integer boundaries

Clip to [0, 10]

Round to 1 decimal place

7. Submission Generation

The final CSV is generated as:

submission_g.csv

With columns:

ID, score

### Running the Code
Requirements

Install dependencies:

pip install torch sentence-transformers scikit-learn pandas numpy

Execution

Make sure your folder structure is:

DATA_DIR/
  ├── train_data.json
  ├── test_data.json
  ├── sample_submission.csv
  └── final_code.py

Then run:

python final_code.py

### Final Output

At the end of execution, the script prints:

Mean and standard deviation of predictions

Count of low / mid / high score predictions

And saves:

submission_g.csv

### Additional Notes

The model is optimized for small GPU usage (MiniLM).

Pseudo labeling helps counter score imbalance.

Noise injection prevents over-smoothing around integer scores.

The approach avoids using any external LLMs, following competition rules.
