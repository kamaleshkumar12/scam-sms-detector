# ScamShield: Scam SMS Detector (Tanglish-aware, classical ML)

Classical-ML (no deep learning) classifier for scam and spam SMS: fake KYC, UPI, courier, loan, lottery and job scams. Includes a normalizer for obfuscation tricks like `bl0cked`, `K.Y.C`, zero-width characters and homoglyphs, and supports romanized Tamil-English (Tanglish) text, trained mostly on synthetic examples.

**Live demo:** https://scam-sms-detector-1.onrender.com (free instance, first load takes about a minute)

**Model:** https://huggingface.co/Kamaleshkumar07/scam-sms-detector

## Problem
Spam filters trained on Western data can miss Indian-style scams. A false positive (blocking a legit bank OTP) is costly, so the decision threshold is set for about 1% false positives on legit messages.

## Approach
- Text normalizer: leetspeak, zero-width characters, homoglyphs, dotted/spaced letters, URL and phone masking
- Features: char n-gram TF-IDF + word TF-IDF + 17 handcrafted features (URLs, shorteners, suspicious TLDs, urgency words, Tamil-script ratio, etc.)
- Models: Logistic Regression, LinearSVC, NB-SVM, SGD, LightGBM (Optuna-tuned, grouped 5-fold CV)
- Stacking meta-learner + Platt calibration
- Threshold chosen on out-of-fold scores for about 1% FPR
- Adversarial training: obfuscated copies of spam and legit messages added to training data
- Group-aware split to prevent duplicate and template leakage

## Data
- UCI SMS Spam Collection (UK English)
- Synthetic Tanglish / Indian-style templates (training help only)
- Own real messages: only 4 (anonymized)

## Results (held-out test, 745 messages, group-deduplicated)
| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 99.2% | about 98.4 to 99.7 |
| ROC-AUC | 0.9986 | 0.9967 to 0.9998 |
| PR-AUC | 0.9937 | 0.9859 to 0.9992 |
| Recall | 98.2% (111/113) | 95.6 to 100 |
| False positive rate | 0.63% (4/632) | 0.16 to 1.27 |
| Precision | 96.5% | 93.2 to 99.1 |
| MCC | 0.969 | - |

**Important:** 731 of the 745 test messages come from the UCI dataset (UK English). Only 1 test message is a real Indian message, so these numbers reflect UCI-style spam, not Indian scam traffic. Indian-message performance is unmeasured.

## Limitations
- Scope: SMS text spam / scam classification only (no link checking, no transaction fraud, no voice or image scams)
- Real Indian data: only 4 messages in total, so Indian-specific accuracy is unknown
- Tanglish training data is mostly synthetic templates; unseen Tanglish wording may be missed
- Tanglish (romanized) is supported; Tamil-script messages are not covered by training data
- Obfuscation robustness is partial; adversarial training covered known tricks only
- Bare-number messages (e.g. "12345") can be falsely flagged
- Legit promotional or urgent-sounding messages may be flagged as spam
- Threshold comes from out-of-fold scores, so 1% FPR is not guaranteed
- Scammers change tactics; periodic retraining is needed
- Do not use as the only filter for blocking bank or OTP messages

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py`: Streamlit demo
- `scam_detector.py`: normalizer, features, prediction code
- `scam_model.joblib`: trained model (4 MB)
- `training.ipynb`: full training pipeline

## Author
Kamaleshkumar K

## License
MIT
