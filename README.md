# 📘 Citation Prediction with Machine Learning
This is the machine learning challenge I had during "Machine learning" course in Tilburg, data science and society program.
> 🎉 In this machine learning challenge, I ranked **15th out of 150 students** by building an accurate and optimized citation prediction model. The ranking criteria was the lowest MAE.

This project focuses on predicting the number of citations (`n_citation`) a research paper will receive based on its metadata. The goal is to build and evaluate several machine learning models using both structured and unstructured features, including text fields like `title`, `abstract`, and `authors`.

---

## 📂 Dataset

- `train.json`: Contains labeled data with citation counts.
- `test.json`: Test data for inference (without citation labels).
- Each record includes:
  - `title` (text)
  - `abstract` (text)
  - `authors` (text)
  - `year` (numeric)
  - `references` (list of paper IDs)

---

## 🧹 Data Preprocessing

- Created a new feature `times_of_references` by counting the number of references in each paper.
- Dropped irrelevant fields such as `references` and `id`.
- Retained text fields for later TF-IDF vectorization.

---

## 🏗️ Feature Engineering

Used `ColumnTransformer` to process different types of input features:

- **Numerical features**:
  - `year`
  - `times_of_references`
- **Text features** (vectorized using TF-IDF):
  - `abstract` → max 4000 features
  - `title` → max 1000 features
  - `authors` → max 1000 features

---

## 🔍 Models Used & Optimization

### ✅ LightGBM (Main Model)

- Built using a full pipeline including preprocessing and regression.
- Target variable (`n_citation`) was log-transformed using `log1p` to reduce skewness.
- Hyperparameters were optimized using **Grid Search**, focusing on:
  - `num_leaves`
  - `n_estimators`
  - `max_depth`
  - `learning_rate`

