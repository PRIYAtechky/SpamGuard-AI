"""
SpamGuard AI - Standalone Model Training Pipeline
Reuses the CountVectorizer and Multinomial Naive Bayes architecture
from the original Mail.ipynb notebook.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.pipeline import Pipeline


def train_and_save_model(data_path=None, model_dir=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if data_path is None:
        candidates = [
            os.path.join(base_dir, 'mail_data.csv'),
            os.path.join(os.getcwd(), 'mail_data.csv'),
        ]
        for c in candidates:
            if os.path.exists(c):
                data_path = c
                break
                
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset 'mail_data.csv' not found. Searched paths: {candidates}")

    if model_dir is None:
        model_dir = os.path.join(base_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    print(f"[*] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # 1. Dataset stats
    total_samples = len(df)
    duplicates_count = int(df.duplicated(subset=['Message']).sum())
    missing_count = int(df.isnull().sum().sum())

    # Handle nulls if present
    data = df.where((pd.notnull(df)), '')

    # Map labels: ham -> 0, spam -> 1
    # Check current label format
    if 'Category' in data.columns and 'Message' in data.columns:
        cat_col = 'Category'
        msg_col = 'Message'
    else:
        # Fallback to column indices
        cat_col = data.columns[0]
        msg_col = data.columns[1]

    ham_count = int((data[cat_col].astype(str).str.lower() == 'ham').sum())
    spam_count = int((data[cat_col].astype(str).str.lower() == 'spam').sum())

    print(f"[*] Total dataset samples: {total_samples}")
    print(f"    - Ham count: {ham_count} ({ham_count/total_samples*100:.2f}%)")
    print(f"    - Spam count: {spam_count} ({spam_count/total_samples*100:.2f}%)")
    print(f"    - Duplicate messages: {duplicates_count}")
    print(f"    - Missing values: {missing_count}")

    X = data[msg_col]
    Y = data[cat_col].map(lambda x: 'spam' if str(x).lower().strip() == 'spam' else 'ham')

    # 2. Train-test split (exact methodology from Mail.ipynb: test_size=0.2, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print(f"[*] Split dataset: {len(X_train)} training samples, {len(X_test)} testing samples")

    # 3. Pipeline creation: CountVectorizer + MultinomialNB
    # In Mail.ipynb: feature_extraction = CountVectorizer(), model = MultinomialNB()
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    print("[*] Training CountVectorizer + MultinomialNB pipeline...")
    pipeline.fit(X_train, Y_train)

    # 4. Model Evaluation on Holdout Test Set
    Y_pred = pipeline.predict(X_test)
    Y_pred_proba = pipeline.predict_proba(X_test)

    acc = float(accuracy_score(Y_test, Y_pred))
    prec_macro = float(precision_score(Y_test, Y_pred, average='macro'))
    rec_macro = float(recall_score(Y_test, Y_pred, average='macro'))
    f1_macro = float(f1_score(Y_test, Y_pred, average='macro'))

    prec_ham = float(precision_score(Y_test, Y_pred, pos_label='ham'))
    rec_ham = float(recall_score(Y_test, Y_pred, pos_label='ham'))
    f1_ham = float(f1_score(Y_test, Y_pred, pos_label='ham'))

    prec_spam = float(precision_score(Y_test, Y_pred, pos_label='spam'))
    rec_spam = float(recall_score(Y_test, Y_pred, pos_label='spam'))
    f1_spam = float(f1_score(Y_test, Y_pred, pos_label='spam'))

    cm = confusion_matrix(Y_test, Y_pred, labels=['ham', 'spam'])
    tn, fp, fn, tp = cm.ravel()

    # Extract vocabulary size
    vocab_size = len(pipeline.named_steps['vectorizer'].vocabulary_)

    metrics = {
        'model_name': 'Multinomial Naive Bayes',
        'vectorizer': 'CountVectorizer (Word Bag-of-Words)',
        'dataset': {
            'total_samples': total_samples,
            'ham_samples': ham_count,
            'spam_samples': spam_count,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'duplicates_count': duplicates_count,
            'missing_values': missing_count,
            'vocabulary_size': vocab_size,
            'avg_ham_length': round(float(data[data[cat_col].astype(str).str.lower() == 'ham'][msg_col].str.len().mean()), 2),
            'avg_spam_length': round(float(data[data[cat_col].astype(str).str.lower() == 'spam'][msg_col].str.len().mean()), 2)
        },
        'evaluation': {
            'accuracy': round(acc, 4),
            'accuracy_percentage': round(acc * 100, 2),
            'macro_avg': {
                'precision': round(prec_macro, 4),
                'recall': round(rec_macro, 4),
                'f1_score': round(f1_macro, 4)
            },
            'weighted_avg': {
                'precision': round(float(precision_score(Y_test, Y_pred, average='weighted')), 4),
                'recall': round(float(recall_score(Y_test, Y_pred, average='weighted')), 4),
                'f1_score': round(float(f1_score(Y_test, Y_pred, average='weighted')), 4)
            },
            'classes': {
                'ham': {
                    'precision': round(prec_ham, 4),
                    'recall': round(rec_ham, 4),
                    'f1_score': round(f1_ham, 4),
                    'support': int((Y_test == 'ham').sum())
                },
                'spam': {
                    'precision': round(prec_spam, 4),
                    'recall': round(rec_spam, 4),
                    'f1_score': round(f1_spam, 4),
                    'support': int((Y_test == 'spam').sum())
                }
            },
            'confusion_matrix': {
                'labels': ['ham', 'spam'],
                'matrix': cm.tolist(),
                'true_negative_ham': int(tn),
                'false_positive_spam': int(fp),
                'false_negative_ham': int(fn),
                'true_positive_spam': int(tp)
            }
        }
    }

    print("\n" + "="*50)
    print("      EVALUATION METRICS (Holdout Test Set)")
    print("="*50)
    print(f"Accuracy:           {metrics['evaluation']['accuracy_percentage']}%")
    print(f"Spam Precision:     {metrics['evaluation']['classes']['spam']['precision']*100:.2f}%")
    print(f"Spam Recall:        {metrics['evaluation']['classes']['spam']['recall']*100:.2f}%")
    print(f"Spam F1-Score:      {metrics['evaluation']['classes']['spam']['f1_score']:.4f}")
    print(f"Ham F1-Score:       {metrics['evaluation']['classes']['ham']['f1_score']:.4f}")
    print(f"Confusion Matrix:   [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
    print("="*50 + "\n")

    # 5. Serialize pipeline and metrics
    model_path = os.path.join(model_dir, 'spam_model.pkl')
    metrics_path = os.path.join(model_dir, 'metrics.json')

    joblib.dump(pipeline, model_path)
    print(f"[+] Saved trained Scikit-learn Pipeline to: {model_path}")

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Saved evaluation metrics to: {metrics_path}")

    return pipeline, metrics


if __name__ == '__main__':
    train_and_save_model()
