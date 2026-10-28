# Sentinel — Credit Card Fraud Detection

Streamlit demo using a class-weighted Random Forest on a real, highly imbalanced public dataset.

## Dataset
- Public CSV mirror: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
- Original dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284,807 historical transactions, including 492 fraud cases.
- `Class=1` indicates fraud; `Class=0` indicates legitimate.
- `V1`–`V28` are anonymized PCA features; `Time` and `Amount` are included.

## Run locally
Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Visit https://share.streamlit.io/ and sign in.
3. Create an app, choose your repository and branch, and set the main file to `app.py`.
4. Deploy; first startup downloads the dataset.

Guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Model and evaluation
- Stratified train/test split.
- Median imputation and standardization in a pipeline.
- Class-weighted Random Forest.
- Reports ROC-AUC, average precision (PR-AUC), precision, recall, F1, confusion matrix, classification report, and a threshold curve.
- Accuracy is not emphasized because fraud is rare.
- The sample scoring form initializes V1–V28 from an example row. Real scoring requires valid feature values generated with the same feature pipeline.

## Limitations
Historical anonymized data from a short period is not representative of every payment environment. Random holdout can leak temporal patterns; use time-based validation for rigorous evaluation. A score is not proof of fraud and must not be the sole basis for blocking transactions. Do not upload private cardholder data.
