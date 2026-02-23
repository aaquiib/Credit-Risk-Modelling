import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Analyser",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ---- global ---- */
        html, body, [data-testid="stAppViewContainer"] {
            background: #0f1117;
            color: #e6eaf0;
            font-family: 'Segoe UI', sans-serif;
        }
        [data-testid="stHeader"] { background: transparent; }

        /* ---- card ---- */
        .card {
            background: #1a1d27;
            border: 1px solid #2a2d3e;
            border-radius: 16px;
            padding: 28px 32px;
            margin-bottom: 20px;
        }
        .card-title {
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: 18px;
        }

        /* ---- hero ---- */
        .hero {
            text-align: center;
            padding: 48px 24px 32px;
        }
        .hero h1 {
            font-size: 2.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .hero p {
            color: #9ca3af;
            font-size: 1rem;
            max-width: 520px;
            margin: 0 auto;
        }

        /* ---- risk badge ---- */
        .risk-good {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border-radius: 14px;
            padding: 32px;
            text-align: center;
            color: white;
        }
        .risk-bad {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            border-radius: 14px;
            padding: 32px;
            text-align: center;
            color: white;
        }
        .risk-label {
            font-size: 1.1rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            opacity: 0.9;
        }
        .risk-value {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 6px 0;
        }
        .risk-sub {
            font-size: 0.9rem;
            opacity: 0.8;
        }

        /* ---- progress bar ---- */
        .prob-bar-bg {
            background: #2a2d3e;
            border-radius: 999px;
            height: 10px;
            overflow: hidden;
            margin-top: 6px;
        }
        .prob-bar-fill-good {
            background: linear-gradient(90deg, #10b981, #34d399);
            height: 10px;
            border-radius: 999px;
            transition: width 0.6s ease;
        }
        .prob-bar-fill-bad {
            background: linear-gradient(90deg, #ef4444, #f87171);
            height: 10px;
            border-radius: 999px;
            transition: width 0.6s ease;
        }

        /* ---- factor chip ---- */
        .chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
        .chip {
            background: #2a2d3e;
            border: 1px solid #3a3d52;
            border-radius: 999px;
            padding: 4px 14px;
            font-size: 0.8rem;
            color: #c4c9d8;
        }
        .chip-warn {
            background: #3b1f1f;
            border: 1px solid #7f1d1d;
            color: #fca5a5;
        }
        .chip-ok {
            background: #1a2e25;
            border: 1px solid #065f46;
            color: #6ee7b7;
        }

        /* ---- divider ---- */
        .divider {
            border: none;
            border-top: 1px solid #2a2d3e;
            margin: 20px 0;
        }

        /* ---- button ---- */
        div.stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #6366f1 0%, #a78bfa 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 14px 0;
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        div.stButton > button:hover { opacity: 0.88; }

        /* ---- selectbox & number inputs ---- */
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input {
            background: #252836 !important;
            border: 1px solid #3a3d52 !important;
            border-radius: 10px !important;
            color: #e6eaf0 !important;
        }
        label { color: #9ca3af !important; font-size: 0.88rem !important; }

        /* remove default streamlit padding top */
        .block-container { padding-top: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("notebook/saved_models/Xgboost_model1.joblib")

model = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🏦 Credit Risk Analyser</h1>
        <p>Fill in the applicant details below to get an instant AI-powered credit risk assessment.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Layout: form (left) | result (right) ─────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    # ── Personal details ──────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Personal Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    with c2:
        sex = st.selectbox("Sex", ["female", "male"])

    job_map = {
        "Skilled": "2",
        "Highly skilled": "3",
        "Unskilled (resident)": "1",
        "Unskilled (non-resident)": "0",
    }
    job_label = st.selectbox("Job / Skill Level", list(job_map.keys()), index=0)
    job = job_map[job_label]

    housing = st.selectbox("Housing", ["own", "free", "rent"])
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Financial details ─────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">Financial Details</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        credit_amount = st.number_input("Credit Amount (DM)", min_value=100, max_value=200_000, value=4000, step=100)
    with c4:
        duration = st.number_input("Loan Duration (months)", min_value=1, max_value=120, value=24, step=1)

    saving_map = {
        "Little  (< 100 DM)": "little",
        "Moderate  (100–500 DM)": "moderate",
        "Quite Rich  (500–1000 DM)": "quite rich",
        "Rich  (> 1000 DM)": "rich",
    }
    saving_label = st.selectbox("Saving Accounts", list(saving_map.keys()))
    saving = saving_map[saving_label]

    checking_map = {
        "Moderate  (0–200 DM)": "moderate",
        "Little  (< 0 DM)": "little",
        "Rich  (> 200 DM)": "rich",
    }
    checking_label = st.selectbox("Checking Account", list(checking_map.keys()))
    checking = checking_map[checking_label]

    purpose = st.selectbox(
        "Purpose of Loan",
        ["radio/TV", "furniture/equipment", "car", "business",
         "domestic appliances", "repairs", "vacation/others", "education"],
    )
    st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("Analyse Credit Risk")

# ── Result panel ──────────────────────────────────────────────────────────────
with right:
    if predict_btn:
        input_df = pd.DataFrame([{
            "Age": age,
            "Credit amount": credit_amount,
            "Duration": duration,
            "Sex": sex,
            "Job": job,
            "Housing": housing,
            "Saving accounts": saving,
            "Checking account": checking,
            "Purpose": purpose,
        }])

        prediction = model.predict(input_df)[0]          # 0 = bad, 1 = good
        proba = model.predict_proba(input_df)[0]          # [p_bad, p_good]
        p_good = float(proba[1])
        p_bad  = float(proba[0])

        is_good = prediction == 1
        risk_css = "risk-good" if is_good else "risk-bad"
        risk_word = "Low Risk" if is_good else "High Risk"
        risk_icon = "✅" if is_good else "⚠️"
        main_pct = f"{p_good*100:.1f}%" if is_good else f"{p_bad*100:.1f}%"
        main_label = "Good Credit Probability" if is_good else "Bad Credit Probability"

        st.markdown(
            f"""
            <div class="{risk_css}">
                <div class="risk-label">{risk_icon} {risk_word}</div>
                <div class="risk-value">{main_pct}</div>
                <div class="risk-sub">{main_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Probability breakdown ─────────────────────────────────────────────
        st.markdown('<div class="card" style="margin-top:18px">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Probability Breakdown</div>', unsafe_allow_html=True)

        st.markdown(f"**Good Credit** &nbsp; `{p_good*100:.1f}%`", unsafe_allow_html=True)
        st.markdown(
            f'<div class="prob-bar-bg"><div class="prob-bar-fill-good" style="width:{p_good*100:.1f}%"></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(f"<br>**Bad Credit** &nbsp; `{p_bad*100:.1f}%`", unsafe_allow_html=True)
        st.markdown(
            f'<div class="prob-bar-bg"><div class="prob-bar-fill-bad" style="width:{p_bad*100:.1f}%"></div></div>',
            unsafe_allow_html=True,
        )

        # ── Risk factor chips ─────────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Key Factors</div>', unsafe_allow_html=True)

        chips_html = '<div class="chip-row">'

        def chip(label, warn=False):
            cls = "chip chip-warn" if warn else "chip chip-ok"
            return f'<span class="{cls}">{label}</span>'

        chips_html += chip(f"Duration: {duration}mo", warn=duration > 36)
        chips_html += chip(f"Amount: {credit_amount:,} DM", warn=credit_amount > 10_000)
        chips_html += chip(f"Age: {age}", warn=age < 25)
        chips_html += chip(f"Savings: {saving}", warn=saving == "little")
        chips_html += chip(f"Checking: {checking}", warn=checking == "little")
        chips_html += chip(f"Housing: {housing}", warn=housing == "rent")
        high_risk_purposes = {"vacation/others", "repairs", "domestic appliances"}
        chips_html += chip(f"Purpose: {purpose}", warn=purpose in high_risk_purposes)

        chips_html += "</div>"
        st.markdown(chips_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Recommendation ────────────────────────────────────────────────────
        if is_good:
            rec_color = "#10b981"
            rec_icon = "✅"
            if p_good >= 0.80:
                rec_text = "Strong credit profile. Loan approval recommended."
            elif p_good >= 0.65:
                rec_text = "Acceptable credit profile. Approval with standard terms."
            else:
                rec_text = "Borderline profile. Consider additional verification."
        else:
            rec_color = "#ef4444"
            rec_icon = "⚠️"
            if p_bad >= 0.80:
                rec_text = "High default risk. Loan not recommended."
            elif p_bad >= 0.65:
                rec_text = "Elevated risk. Requires collateral or co-signer."
            else:
                rec_text = "Moderate-high risk. Manual review advised."

        st.markdown(
            f"""
            <div style="border-left: 4px solid {rec_color}; padding: 14px 20px;
                        background:#1a1d27; border-radius: 0 12px 12px 0; margin-top:4px">
                <span style="font-size:1rem">{rec_icon} <b style="color:{rec_color}">Recommendation</b></span><br>
                <span style="color:#c4c9d8; font-size:0.92rem">{rec_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        # Placeholder when no prediction yet
        st.markdown(
            """
            <div class="card" style="text-align:center; padding: 60px 32px;">
                <div style="font-size:3rem; margin-bottom:16px">🔍</div>
                <div style="font-size:1.1rem; font-weight:600; color:#c4c9d8">
                    Fill in the form and click<br><b>Analyse Credit Risk</b>
                </div>
                <div style="color:#6b7280; font-size:0.88rem; margin-top:10px">
                    Results will appear here — including risk score,<br>
                    probability breakdown, and recommendation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; color:#4b5563; font-size:0.78rem; padding: 28px 0 12px">
        Powered by XGBoost · German Credit Dataset · Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
