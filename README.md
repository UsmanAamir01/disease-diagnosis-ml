# 🧬 TF-IDF vs One-Hot: Dimensionality Reduction for Disease Classification

This project uses machine learning to explore how different textual feature encoding methods — **TF-IDF** and **One-Hot Encoding** — impact disease classification. Complete with data processing, visualization, classification models, and an interactive **Streamlit dashboard**.

---

## 📦 Table of Contents

- [1. Overview](#1-overview)
- [2. Features](#2-features)
- [3. Dataset](#3-dataset)
- [4. Installation](#4-installation)
- [5. Usage](#5-usage)
- [6. Results](#6-results)
- [7. Streamlit Dashboard](#7-streamlit-dashboard)
- [8. Limitations & Future Work](#8-limitations--future-work)
- [9. Contributing](#9-contributing)
- [10. License](#10-license)

---

## 1. 🧠 Overview

In healthcare analytics, extracting meaningful signals from textual annotations (e.g., symptoms, risk factors) is vital. This project compares:

- **TF-IDF vs One-Hot** encoding
- Dimensionality reduction via **PCA** and **Truncated SVD**
- Classification via **K-Nearest Neighbors (KNN)** and **Logistic Regression**

The app allows real-time exploration of these techniques using an intuitive Streamlit interface.

---

## 2. ✨ Features

- 📁 Text preprocessing and encoding (TF-IDF and One-Hot)
- 📉 Dimensionality reduction using PCA and SVD
- 📊 Classifier benchmarking (KNN and Logistic Regression)
- 🧪 Safe cross-validation with class-scarce datasets
- 📺 Interactive Streamlit dashboard with plots, model tuning, and metrics
- 📁 Exportable results in CSV format

---

## 3. 📚 Dataset

### 3.1 Input Files
- `disease_features.csv`: Contains columns — Disease, Risk Factors, Symptoms, Signs, and Category.
- `encoded_output2.csv`: Precomputed One-Hot encoded features (binary format).

### 3.2 Text Preprocessing Example
```python
for col in ['Risk Factors', 'Symptoms', 'Signs']:
    features_df[col] = (
        features_df[col]
        .apply(ast.literal_eval)
        .apply(lambda lst: ' '.join(lst))
    )
```

---

## 4. ⚙️ Installation

### 4.1 Requirements
- Python ≥ 3.7
- pandas
- scikit-learn
- matplotlib
- streamlit

### 4.2 Install Dependencies
```bash
pip install pandas scikit-learn streamlit matplotlib
```

---

## 5. 🚀 Usage

To launch the Streamlit dashboard:
```bash
streamlit run disease_prediction_streamlit.py
```

### 5.1 Dashboard Features
- 🔘 Select feature encoding: TF-IDF or One-Hot
- ⚙️ Configure model parameters:
  - KNN (k, distance metric)
  - Logistic Regression
- 📊 Visualize 2D embeddings (PCA/SVD)
- 📈 Evaluate and compare performance metrics

---

## 6. 📈 Results

Results show that:
- **Logistic Regression** achieves perfect training scores (indicative of overfitting).
- **KNN** performs better with One-Hot encoding than TF-IDF (F1 ≈ 0.33 vs. 0.12).
- **TF-IDF** provides richer clusters for category separation but may underperform in classification.

---

## 7. 📺 Streamlit Dashboard

The interactive app includes:
- **Encoding Selection**: Radio buttons (TF-IDF / One-Hot)
- **Model Config**: Sliders and dropdowns for KNN and Logistic Regression
- **Cross-Validation**: Dynamic fallback if stratification fails
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Visualizations**: PCA/SVD scatter plots, heatmaps, bar charts
- **Export Options**: Downloadable results as CSV

---

## 8. 🧪 Limitations & Future Work

- ⚠️ Each disease class has only one sample → no stratified k-fold possible
- 🧬 Needs more samples per disease for generalizable evaluation
- 🔄 Future improvements:
  - Incorporate deep learning embeddings (e.g., BERT)
  - Expand dataset
  - Deploy the dashboard publicly

---

## 9. 🤝 Contributing

Feel free to fork the repo and open PRs! Suggestions and improvements are welcome.

---

## 10. 📄 License

This project is intended for academic and educational use. Licensing terms to be added.

---
