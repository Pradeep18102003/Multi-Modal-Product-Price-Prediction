# Multi-Modal Product Price Prediction

**Predict the price of a product using both images and text descriptions through a multi-modal machine learning pipeline.**

---

## 📁 Project Structure
```
📂 Multi-Modal Product Price Prediction
│── 📂 notebook
│   └── multi-modal-product-price-prediction.ipynb
│── README.md  # Project documentation

```
---


---

## 📝 Overview

This repository showcases a robust multi-modal approach for predicting product prices with:

- **Image Input:** Features via ResNet50 (pretrained on ImageNet)
- **Text Input:** Features via SentenceTransformer (`all-MiniLM-L6-v2`)
- **Feature Engineering:** Quantity, pack count, description length, organic flag, brand (categorically encoded)

All features are concatenated for training in gradient-boosted regression frameworks.

---

## 📊 Data

- Product descriptions (text)
- Product images (Amazon-style URLs)
- Ground-truth prices (log1p-transformed for regression)

---

## 🚀 Model Workflow

1. **Process Images:** Extract 2048D vector from ResNet50 for each product image
2. **Process Text:** Extract 384D vector from SentenceTransformer for each product description
3. **Feature Engineering:** Extract quantity, pack count, text length, organic flag, brand (with missing value handling and categorical encoding)
4. **Final Feature Set:** Concatenate image, text, and engineered features (total features: 2437 per product)
5. **Model Benchmarking:** Train & evaluate LightGBM, XGBoost, CatBoost using 20% hold-out validation on log1p price, report **SMAPE** (Symmetric Mean Absolute Percentage Error)

---

## 🏆 Benchmarking Results

| Model      | Validation SMAPE (%)  | Training Time (sec) |
|------------|----------------------|---------------------|
| LightGBM   | **55.24**            | 469.42              |
| CatBoost   | 56.13                | 468.35              |
| XGBoost    | 58.49                | 298.68              |

*Best performing model: **LightGBM** with lowest SMAPE and competitive runtime.*

---

## 🔧 Technical Highlights

- Fully parallelized image downloading and feature extraction (multiprocessing)
- GPU-enabled text and image embedding (ResNet50, SentenceTransformer)
- Extensive feature engineering (quantity extraction, pack parsing, brand encoding, robust missing value handling)
- Unified feature matrix for all models (engineering + embeddings)
- Automatic benchmarking loop for model selection

---

## ▶️ How To Use

1. Place the train/test CSVs and product images as directed in the notebook cells
2. Run `multi-modal-product-price-prediction.ipynb` in any compatible notebook environment (Jupyter, Kaggle, Colab) with GPU
3. Review the final benchmarking summary for model choice and performance

---

## 📚 References

- [ResNet50 Paper](https://arxiv.org/abs/1512.03385)
- [Sentence Transformers](https://www.sbert.net/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

---

**Maintained by Pradeep Kumar (IIT BHU, Data Science Student).**

---


