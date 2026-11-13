# SpamGuard AI – Standalone Email Spam Detection System

[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit: 1.48+](https://img.shields.io/badge/Streamlit-1.48+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-MultinomialNB-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Accuracy: 99.19%](https://img.shields.io/badge/Accuracy-99.19%25-brightgreen.svg)]()
[![Zero Network Overhead](https://img.shields.io/badge/Architecture-Standalone%20Python-blue.svg)]()

> A standalone, high-performance cybersecurity web application built with **Streamlit** and **Scikit-learn**. Performs instant email spam and phishing detection locally in Python—**with no Flask backend, no REST APIs, and no external AI API dependencies**.

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │        Browser / User Interface        │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │      Streamlit Standalone Engine       │
                      │               (app.py)                 │
                      └───────────┬────────────────┬───────────┘
                                  │                │
            Direct In-Memory Load │                │ Local SQLite Queries
                                  ▼                ▼
         ┌────────────────────────────────┐  ┌──────────────────────┐
         │  Scikit-Learn Pipeline (.pkl)  │  │   SQLite Database    │
         │  CountVectorizer + MNB Model   │  │   (spam_guard.db)    │
         │  (Sub-millisecond inference)   │  │  (Sanitized previews)│
         └────────────────────────────────┘  └──────────────────────┘
```

### Why Standalone?
- **Zero API Latency:** Predictions run directly inside the Python process in $< 1\text{ ms}$, eliminating HTTP network round-trips.
- **Privacy by Design:** Email text is analyzed on your local machine. No text is sent to third-party APIs or cloud endpoints. Full email bodies are never stored; only sanitized 140-character previews are recorded in SQLite.
- **Zero Configuration:** No need to run or coordinate separate frontend and backend servers. Just run `streamlit run app.py`.

---

## Key Features

1. **Email Spam & Threat Detection**:
   - Analyzes raw message text using the pre-trained `model/spam_model.pkl` pipeline.
   - 5 built-in preset scenarios for instant demonstration (Lottery Scam, Bank Fraud, Work Sync, Casual Dinner, Handset Promo).
   - Real-time word and character counters with 100KB payload safety limiter.
   - Clean SaaS-style verdict banners:
     - 🚨 **SPAM / PHISHING DETECTED** (rose banner) with threat explanation.
     - ✅ **VERIFIED LEGITIMATE (HAM)** (emerald banner) with safe communication profile.
   - Dual class probability distribution bars (Ham % vs. Spam %) and inference latency meter.

2. **Real-Time Analytics Dashboard**:
   - Aggregate statistics calculated directly from the local SQLite database (`spam_guard.db`):
     - Total emails analyzed
     - Spam threats detected
     - Legitimate emails (Ham)
     - Spam detection percentage
   - Interactive SVG **Donut Chart** (`Altair`) for class distribution.
   - **Chronological Detection Timeline Chart** (`Altair`) tracking daily volume.
   - Recent detections table preview.

3. **Prediction Audit History**:
   - Paginated table of past scans with sanitized previews, verdict status badges, confidence ratings, and timestamps.
   - Text search filter and class dropdown filter (`All`, `Spam Only`, `Ham Only`).
   - "Clear Audit History" workflow with safety confirmation check.

4. **Transparent Model Architecture**:
   - Holdout test set metrics computed on 1,115 unobserved messages:
     - **Accuracy:** `99.19%`
     - **Spam Precision:** `100.00%` (0% false alarm rate)
     - **Spam Recall:** `93.96%`
     - **Vocabulary Size:** `7,701` features
   - Interactive 2x2 **Confusion Matrix** visual table.
   - Scikit-learn Classification Report breakdown (Ham vs. Spam).
   - Mathematical formulation for Multinomial Naive Bayes with Laplace smoothing ($α = 1.0$).
   - Explicitly documented model boundaries and limitations.

5. **About Project View**:
   - System architecture diagram, problem & solution background, complete technology stack table, and disclaimers.

---

## Machine Learning Model Evaluation

The model was trained on the 5,572-message Kaggle/UCI spam dataset (`mail_data.csv`) using an 80/20 train/test split (`random_state=42`).

### Holdout Test Metrics (Unobserved Data)

| Metric | Overall | Ham (Legitimate) | Spam (Malicious) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **99.19%** | — | — |
| **Precision** | — | 99.08% | **100.00%** |
| **Recall** | — | **100.00%** | 93.96% |
| **F1-Score** | — | **0.9954** | **0.9689** |
| **Support** | 1,115 | 966 | 149 |

### Confusion Matrix
```
                    Predicted HAM    Predicted SPAM
Actual HAM (966)         966               0       <-- 0% False Positives
Actual SPAM (149)          9             140       <-- 93.96% True Positives
```

---

## Quick Start (Run Locally)

### 1. Prerequisites
- Python 3.10 or higher.
- `pip` package manager.

### 2. Installation
Open your terminal in the project directory:
```bash
# 1. (Optional) Create and activate a virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python test_app.py
```
*Runs 7 unit tests verifying model loading, direct inference, sanitization, and SQLite operations.*

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```
The application will launch automatically in your default browser at:
`http://localhost:8501`

---

## Project Structure

```
spam-email-detection/
│
├── app.py                   # Main standalone Streamlit web application
├── database.py              # SQLite database manager (sanitized previews, audit history & stats)
├── train_model.py           # Reproducible model training script (saves to model/)
├── test_app.py              # Automated unit and integration test suite (7/7 tests passing)
│
├── model/
│   ├── spam_model.pkl       # Pre-trained Scikit-learn Pipeline (CountVectorizer + MultinomialNB)
│   └── metrics.json         # Evaluated holdout metrics and dataset profile
│
├── mail_data.csv            # Original dataset (5,572 labeled text samples)
├── Mail.ipynb               # Original exploratory data analysis notebook
├── requirements.txt         # Python dependencies (Streamlit, Scikit-learn, Pandas, Altair, Joblib)
├── .gitignore               # Git exclusions for Python cache, DBs, and legacy artifacts
└── README.md                # Project documentation
```

---

## Privacy & Security

- **No Remote Transmission:** All message tokens and classifications are processed locally on the Python host machine.
- **Sanitized Previews Only:** Raw email text is never persisted to disk. Only 140-character sanitized previews (HTML stripped, whitespace normalized) are saved to SQLite.
- **Payload Boundary:** Inputs exceeding 100KB are safely truncated to protect against memory exhaustion.
- **Parameterized SQL:** All SQLite database interactions utilize parameterized queries (`?`), preventing SQL injection attacks.

---

## License

This project is licensed under the [MIT License](LICENSE).
Dataset courtesy of the SMS Spam Collection on UCI Machine Learning Repository / Kaggle.
