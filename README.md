# T.A.M.E. Auditory Pain Analysis for Healthcare Purposes

## Overview

Accurate pain assessment is a critical component of clinical decision-making. With the rise of telehealth services—especially following the COVID-19 pandemic—healthcare providers increasingly rely on remote communication to evaluate patient conditions. In these settings, physicians must depend heavily on self-reported pain levels, which can be subjective and inconsistent.

explores whether a neural network can automatically detect pain from a patient’s voice. The system leverages a pretrained large-scale audio embedding model and fine-tunes it for binary pain classification.

## Project Objective

The primary goal of this project was to:

- Develop a machine learning model capable of predicting whether a speaker is experiencing pain

- Utilize pretrained audio embeddings to improve classification performance

- Evaluate the feasibility of audio-based pain detection in telehealth environments

## Model Architecture

This project builds on top of:

- CLAP (Contrastive Language–Audio Pretraining) from Hugging Face

    - Used as a pretrained audio embedding backbone

    - Extracts high-level semantic audio features

- A custom neural network classification head

    - Fine-tuned on labeled pain vs. no-pain audio samples

    - Trained for binary classification

## Pipeline Overview

1. Audio input preprocessing
2. Feature extraction via CLAP embeddings
3. Dense classification layer
4. Binary prediction: Pain / No Pain

## Model Training

- Dataset: Labeled audio samples (pain vs. no pain)

- Loss Function: Binary Cross-Entropy

- Optimizer: Adam

- Evaluation Metrics:

    - Accuracy

    - Precision

    - Recall

    - F1 Score

    - Confusion Matrix

## Limitations

- Limited dataset size
- Binary classification only (no pain severity estimation)
- Not validated in real-world clinical environments
- Audio quality and background noise may affect performance

## Future Improvements

- Expand dataset with diverse demographics
- Incorporate pain severity regression
- Add noise-robust preprocessing
- Deploy as a telehealth integration prototype
- Perform clinical validation studies