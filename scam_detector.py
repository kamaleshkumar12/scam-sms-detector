import re, unicodedata
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:xyz|top|in|com|co|ly|link|site|online|club|info)\S*|\bbit\.ly/\S+|\bwa\.me/\S+)")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9][\dXx]{9}(?!\d)")
LONG_NUM = re.compile(r"\d{4,}")
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

ZW = re.compile(r"[\u200b-\u200f\u2060\ufeff\u00ad]")
HOMO = {0x0430: "a", 0x0435: "e", 0x043e: "o", 0x0441: "c", 0x0440: "p", 0x0445: "x",
        0x0443: "y", 0x0456: "i", 0x0455: "s", 0x0458: "j",
        0x0410: "A", 0x0415: "E", 0x041e: "O", 0x0421: "C", 0x0420: "P",
        0x0425: "X", 0x0423: "Y", 0x0406: "I"}
SEP_RE = re.compile(r"(?<![a-z0-9])(?:[a-z@$][.\-_]){2,}[a-z@$](?![\w@$/])")
SPC_RE = re.compile(r"(?<![a-z0-9])(?:[a-z@$] ){3,}[a-z@$](?![\w@$/])")

SHORTENERS = ["bit.ly", "tinyurl", "t.co", "goo.gl", "wa.me", "cutt.ly", "rb.gy"]
SUS_TLD = [".xyz", ".top", ".online", ".club", ".site", ".link", ".info"]
URGENT = ["urgent", "immediately", "today", "expire", "block", "suspend", "freez", "deactivat",
          "within 24", "last chance", "udane", "verify", "kyc", "aadhaar", "pan card", "claim",
          "prize", "won", "lottery", "winner", "congrat", "refund", "loan", "earn", "click",
          "pending", "penalty", "legal", "parisu", "install"]
RS_RE = re.compile(r"(?:rs\.?|inr|₹)\s*[\d,]+")


def _deleet(m):
    t = m.group(0)
    if len(re.findall(r"[a-z]", t)) >= 3 and re.search(r"[0-9@$]", t):
        return t.translate(LEET)
    return t


def clean_raw(text):
    t = unicodedata.normalize("NFKC", str(text))
    t = ZW.sub("", t)
    return t.translate(HOMO)


def normalize(text):
    t = clean_raw(text).lower()
    t = t.replace("hxxp", "http").replace("[.]", ".").replace("(.)", ".")
    t = SEP_RE.sub(lambda m: re.sub(r"[.\-_]", "", m.group(0)), t)
    t = SPC_RE.sub(lambda m: m.group(0).replace(" ", ""), t)
    t = URL_RE.sub(" urltoken ", t)
    t = PHONE_RE.sub(" phonetoken ", t)
    t = LONG_NUM.sub(" numtoken ", t)
    t = re.sub(r"[a-z0-9@$]+", _deleet, t)
    return re.sub(r"\s+", " ", t).strip()


def hand_features(raw):
    t = str(raw)
    low = t.lower()
    n = max(len(t), 1)
    words = low.split()
    urls = URL_RE.findall(low)
    return {
        "len": len(t),
        "n_words": len(words),
        "digit_ratio": sum(c.isdigit() for c in t) / n,
        "upper_ratio": sum(c.isupper() for c in t) / n,
        "punct_ratio": sum((not c.isalnum()) and (not c.isspace()) for c in t) / n,
        "n_excl": t.count("!"),
        "n_urls": len(urls),
        "has_shortener": int(any(s in low for s in SHORTENERS)),
        "has_sus_tld": int(any(s in low for s in SUS_TLD)),
        "n_phones": len(PHONE_RE.findall(t)),
        "n_rs": len(RS_RE.findall(low)),
        "urgent_hits": sum(k in low for k in URGENT),
        "tamil_ratio": sum("\u0b80" <= c <= "\u0bff" for c in t) / n,
        "has_otp": int("otp" in low),
        "has_do_not_share": int("do not share" in low or "don't share" in low),
        "has_txn_words": int(any(k in low for k in ["debited", "credited", "upi ref", "a/c xx"])),
        "leet_hits": len(re.findall(r"[a-z]+[0@$][a-z]+", low)),
    }


def build_hand(d):
    return pd.DataFrame([hand_features(clean_raw(x)) for x in d["text"]])


class NBSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, n_text=0, C=4.0, alpha=1.0):
        self.n_text = n_text
        self.C = C
        self.alpha = alpha

    def decision_function(self, X):
        return self.lr_.decision_function(X[:, :self.n_text].multiply(self.r_).tocsr())


def raw_score(m, X):
    if hasattr(m, "decision_function"):
        return m.decision_function(X)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load(path="scam_model.joblib"):
    return joblib.load(path)


def predict_proba(B, texts):
    d = pd.DataFrame({"text": [str(t) for t in texts]})
    d["norm"] = d["text"].map(normalize)
    X = hstack([B["cv"].transform(d["norm"]),
                B["wv"].transform(d["norm"]),
                csr_matrix(B["sc"].transform(build_hand(d)))]).tocsr()
    Z = np.column_stack([raw_score(B["fit"][n], X) for n in B["names"]])
    s = B["meta"].decision_function(Z)
    return B["platt"].predict_proba(s.reshape(-1, 1))[:, 1]


def predict(B, texts):
    p = predict_proba(B, texts)
    return [{"scam_probability": float(x), "is_scam": bool(x > B["thr"])} for x in p]
