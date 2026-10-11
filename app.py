import streamlit as st
import scam_detector as sd

st.set_page_config(page_title="Scam SMS Detector", page_icon="🛡️")

@st.cache_resource
def get_model():
    return sd.load("scam_model.joblib")

B = get_model()

st.title("🛡️ Scam SMS Detector (Tamil / Tanglish / English)")
st.caption("Classical ML: char+word TF-IDF + handcrafted features, stacked ensemble. "
           "Message server la store aagaadhu.")

EXAMPLES = {
    "Fake KYC": "Dear customer your SBI a/c will be blocked today. Update KYC now http://sbi-kyc-update.xyz",
    "Tanglish scam": "Ungal account block aagapogudhu, udane click pannunga bit.ly/x1y2",
    "Obfuscated": "Ur a/c bl0cked. K.Y.C upd@te now hxxp://sbi-kyc[.]xyz",
    "Bank OTP (legit)": "Your OTP for login is 482913. Do not share it with anyone. - HDFC Bank",
    "Friend chat (legit)": "Machi, naalaiku evening 6 mani ku meet pannalam, ok va?",
}
choice = st.selectbox("Example", ["(type my own)"] + list(EXAMPLES))
default = EXAMPLES.get(choice, "")
text = st.text_area("Message", value=default, height=120)

if st.button("Check") and text.strip():
    r = sd.predict(B, [text])[0]
    p = r["scam_probability"]
    if r["is_scam"]:
        st.error(f"⚠️ Likely SCAM  (score {p:.3f})")
    else:
        st.success(f"✅ Looks OK  (score {p:.3f})")
    st.progress(min(max(p, 0.0), 1.0))
    st.caption(f"Decision threshold {B['thr']:.4f} (set for ~1% false positives on legit messages).")
    st.info("Idhu oru ML estimate. Bank / OTP message-a ithai mattum nambi block pannaadhinga.")
