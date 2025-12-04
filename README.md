# SADA - Arabic Dialect Classification using the SADA Dataset
This repository contains a complete machine learning pipeline for Arabic dialect classification using the SADA dataset, developed as part of the IT469 Natural Language Processing course at King Saud University.

The project investigates how well models can generalize across different text domains (Comedy, Drama, Kids) and examines the impact of domain shifts on classification performance.

-------------------------------------------

# Project Overview
Arabic dialect classification is challenging due to variations in vocabulary, pronunciation, and linguistic style across regions.
This project evaluates two models:
- Logistic Regression with TF-IDF features (Baseline)
- Fine-tuned Multilingual BERT (Advanced Model)
Each model is trained on a single domain (e.g., Comedy) and tested on in-domain and out-of-domain data to assess generalization.

# Dataset: SADA (Saudi Audio Dataset for Arabic)
The SADA dataset consists of 253,166 transcribed speech segments from Saudi TV shows (Comedy, Drama, Kids, and others).
The dataset link: https://www.kaggle.com/datasets/sdaiancai/sada2022/data?select=test.csv
- We focus on some of them, as they are the most frequent:
Top 3 dialects: Najdi, Hijazi, Khaliji. 
Top 3 categories: Comedy, Drama, Kids

- To ensure fairness, we created a balanced dataset with:
1676 samples per dialect per category

- Preprocessing steps included:
Removing missing or duplicate rows
Using the ProcessedText field
Normalization (removing diacritics, symbols, emojis)

# Methodology
1- Baseline Model — Logistic Regression
- TF-IDF vectorization (1–4 grams, max 5000 features)
- Multinomial Logistic Regression (lbfgs solver)
- Trained and evaluated per domain

2- Advanced Model — Multilingual BERT
- Pretrained checkpoint: bert-base-multilingual-cased
- Input length: 128 tokens
- Batch size: 16
- Epochs: 10
- Learning rate: 2e-5
- Fine-tuned separately for each domain

3. Cross-Domain Evaluation
For each model:
- Train on Comedy, test on Comedy/Drama/Kids
- Train on Drama, test on Comedy/Drama/Kids
- Train on Kids, test on Comedy/Drama/Kids
This allows us to measure domain shift performance.

# Results Summary
- Logistic Regression
Best performance in Kids → Kids (Accuracy: 49.4%), with a noticeable drop when testing on other domains: Comedy and Drama models achieved ~35–42% accuracy

- BERT
Best performance in Kids → Kids (Accuracy: 47.2%), and similar in-domain performance for Drama and Comedy. However, the Cross-domain performance still drops significantly.

Both models show that domain-specific training strongly influences performance. 
# Statistical Significance: McNemar's Test
To compare in-domain vs. cross-domain models on the same test set:
- Logistic Regression: All p-values < 0.05, meaning statistically significant differences
- BERT: Significant differences in Drama and Kids, not in Comedy
This confirms that domain shifts significantly impact predictions, even when overall accuracy looks similar. So, the challenge lies not in the model's architecture but in the domain's nature.

The complete academic report — including methodology details, figures, mislabeled examples, and McNemar visualizations — is available in: IT469 -SADA Report.pdf.

# Key Takeaways
- Models perform best in the same domain they were trained on.
- Cross-domain generalization is weak, even for BERT.
- Kids' domain leads to better accuracy and transferability due to simpler, more consistent linguistic patterns.
- Dataset noise (mislabeling) affects performance.
