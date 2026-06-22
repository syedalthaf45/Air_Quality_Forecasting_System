# Customer Churn Prediction

**Live demo → [link here]** *(update this)*

---

Built this to predict whether a telecom customer is likely to churn, using the IBM Telco dataset. The goal was to go end-to-end — from raw data to a deployed web app where you can input customer details and get a prediction.

I tried three models (Logistic Regression, Random Forest, Gradient Boosting) and used GridSearchCV to tune each one. Random Forest and Gradient Boosting performed the best. The final model is served through a Flask app deployed on Render.

---

## What it does

- Trains and compares 3 classification models on the IBM Telco churn dataset
- Hyperparameter tuning with GridSearchCV
- Flask web app where you can input customer data and get a churn prediction
- Packaged with Docker and deployed on Render

---

## Dataset

IBM Telco Customer Churn dataset — 7,043 customers, 21 features including tenure, contract type, monthly charges, and internet service details.

---

## Stack

- Python, scikit-learn, Flask
- Docker, Render

---

## Project structure

```
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── model_trainer.py        # trains LR, RF, GB with GridSearchCV
│   ├── model_evaluator.py      # accuracy, F1, ROC-AUC
│   └── utils.py
│
├── templates/
│   ├── index.html              # input form
│   └── result.html             # prediction output
│
├── artifacts/
│   ├── model.pkl               # saved best model
│   └── evaluation_report.json
│
├── app.py
├── run_pipeline.py
├── config.yaml
├── requirements.txt
└── Dockerfile
```

---

## Running locally

```bash
git clone https://github.com/syedalthaf45/customer-churn-prediction.git
cd customer-churn-prediction

pip install -r requirements.txt
```

Train the model:

```bash
python run_pipeline.py
```

Start the app:

```bash
python app.py
```

Open `http://localhost:5000`

---

## Docker

```bash
docker build -t churn-prediction .
docker run -p 5000:5000 churn-prediction
```

---

## Model results

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | ~79% | ~0.76 |
| Random Forest | ~82% | ~0.80 |
| Gradient Boosting | ~83% | ~0.81 |

Full report in `artifacts/evaluation_report.json`.
