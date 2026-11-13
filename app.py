"""
SpamGuard AI - Enterprise Email Spam Detection System
Standalone Streamlit Web Application

Directly executes Scikit-learn MultinomialNB Pipeline without Flask or REST APIs.
Maintains prediction history and analytics in SQLite.
"""

import os
import json
import time
import joblib
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

import database

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM SAAS THEME
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="SpamGuard AI – Email Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS CSS Styling: Light theme, navy headings, lavender/indigo accents, rounded cards
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    /* Card styling */
    .saas-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }

    /* Threat Result Cards */
    .threat-card-spam {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 1.5px solid #fda4af;
        border-radius: 14px;
        padding: 22px;
        color: #881337;
        margin-top: 15px;
    }

    .threat-card-ham {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1.5px solid #86efac;
        border-radius: 14px;
        padding: 22px;
        color: #14532d;
        margin-top: 15px;
    }

    /* Pill Badges */
    .badge-spam {
        background-color: #ffe4e6;
        color: #be123c;
        border: 1px solid #fecdd3;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
        display: inline-block;
        letter-spacing: 0.05em;
    }

    .badge-ham {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
        display: inline-block;
        letter-spacing: 0.05em;
    }

    .badge-subtle {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* Metrics Summary Tiles */
    .kpi-tile {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    .kpi-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 700;
        color: #64748b;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 4px;
    }

    /* Streamlit button tweaks */
    div.stButton > button:first-child {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }

    /* Text area tweak */
    .stTextArea textarea {
        border-radius: 10px;
        border: 1.5px solid #cbd5e1;
        font-size: 0.95rem;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. MODEL LOADER (CACHED FOR INSTANT LOCAL INFERENCE)
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")


@st.cache_resource(show_spinner=False)
def load_detection_model():
    """
    Loads the Scikit-learn CountVectorizer + MultinomialNB pipeline into memory.
    Cached across sessions so inference has zero cold-start delay (<1ms).
    """
    if not os.path.exists(MODEL_PATH):
        # Check fallback path
        fallback = os.path.join(BASE_DIR, "spam_model.pkl")
        if os.path.exists(fallback):
            return joblib.load(fallback)
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_model_metrics():
    """Load verified holdout evaluation metrics from metrics.json."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


pipeline = load_detection_model()
metrics_data = load_model_metrics()

# Preset Email Samples
SAMPLE_EMAILS = [
    {
        "title": "🎁 Lottery Prize Scam",
        "type": "spam",
        "text": "WINNER!! As a valued customer you have been selected to receive a £900 prize reward or brand new camera! To claim call 09061701461 right now. Claim code KL341. Valid 12 hours only. Reply STOP to opt out."
    },
    {
        "title": "🚨 Urgent Bank Security Notice",
        "type": "spam",
        "text": "URGENT! Your Mobile Account Statement indicates 800 unredeemed points and an unverified wire transfer. Please call our urgent automated verification line 08712300220 immediately to secure your balance."
    },
    {
        "title": "💼 Q3 Team Roadmap Sync",
        "type": "ham",
        "text": "Hi Alex, please find attached the revised project roadmap for Q3. Let's discuss our progress during tomorrow's team sync at 10 AM. Let me know if you need any adjustments beforehand."
    },
    {
        "title": "☕ Friendly Dinner Catchup",
        "type": "ham",
        "text": "Hey! Are you still up for catching that documentary tonight? We can grab dinner around 7 PM near the library before the show starts. Let me know if that works for you!"
    },
    {
        "title": "📱 Free Mobile Handset Offer",
        "type": "spam",
        "text": "Had your mobile 11 months or more? You are entitled to Update to the latest color mobile with camera for Free! Call The Mobile Update Co FREE on 08002986030 or text CLAIM."
    }
]


# ------------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & SYSTEM HEALTH
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
        <div style="width: 42px; height: 42px; border-radius: 10px; background: linear-gradient(135deg, #4338ca, #6366f1); display: flex; align-items: center; justify-content: center; color: white; font-size: 22px; box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.3);">
            🛡️
        </div>
        <div>
            <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; line-height: 1.2;">SpamGuard AI</div>
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 500;">Standalone Threat Detection</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if pipeline is not None:
        st.markdown("""
        <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 6px 12px; display: flex; align-items: center; gap: 8px; margin-bottom: 20px;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background-color: #22c55e;"></div>
            <span style="font-size: 0.75rem; font-weight: 600; color: #15803d;">Model Loaded (Local Python)</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 6px 12px; display: flex; align-items: center; gap: 8px; margin-bottom: 20px;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background-color: #ef4444;"></div>
            <span style="font-size: 0.75rem; font-weight: 600; color: #b91c1c;">Model File Missing</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;'>Navigation</p>", unsafe_allow_html=True)

    page = st.radio(
        label="Select Application View",
        options=[
            "🛡️ Email Detection",
            "📊 Analytics Dashboard",
            "📜 Prediction History",
            "🧠 Model Architecture",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #64748b; line-height: 1.6;">
        <strong>Engine:</strong> Multinomial Naive Bayes<br>
        <strong>Vectorizer:</strong> CountVectorizer (BoW)<br>
        <strong>Vocabulary:</strong> 7,701 features<br>
        <strong>Holdout Accuracy:</strong> 99.19%<br>
        <strong>Database:</strong> SQLite3 (Local)<br>
        <strong>Runtime:</strong> Pure Python (No API)
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 4. VIEW: EMAIL DETECTION (MAIN DETECTOR)
# ------------------------------------------------------------------------------
if page == "🛡️ Email Detection":
    st.markdown("""
    <div>
        <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background-color: #eef2ff; border: 1px solid #c7d2fe; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; color: #4338ca; margin-bottom: 10px;">
            ⚡ Local High-Speed Inference
        </div>
        <h1 style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-bottom: 6px; tracking-tight;">
            Email Spam & Phishing Detection
        </h1>
        <p style="font-size: 0.95rem; color: #64748b; margin-bottom: 24px;">
            Inspect suspicious email bodies and messages directly in your browser. Powered by an in-memory Scikit-learn Naive Bayes pipeline with verified 99.19% test accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if pipeline is None:
        st.error("⚠️ The trained model `model/spam_model.pkl` could not be located. Run `python train_model.py` to compile the pipeline.")
        st.stop()

    # Preset Sample Selector
    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #334155; margin-bottom: 8px;'>Load a Test Scenario:</p>", unsafe_allow_html=True)
    preset_cols = st.columns(len(SAMPLE_EMAILS))
    
    if "input_email_text" not in st.session_state:
        st.session_state.input_email_text = ""

    for i, sample in enumerate(SAMPLE_EMAILS):
        if preset_cols[i].button(sample["title"], key=f"preset_{i}", use_container_width=True):
            st.session_state.input_email_text = sample["text"]

    # Main Email Text Area
    email_text = st.text_area(
        label="Email Message Content",
        value=st.session_state.input_email_text,
        height=190,
        placeholder="Paste an email message, header body, or SMS text here to inspect for threat indicators...",
        help="Paste complete message body. Input is validated and processed locally."
    )

    # Word and character count indicator
    char_count = len(email_text)
    word_count = len(email_text.strip().split()) if email_text.strip() else 0

    col_meta1, col_meta2 = st.columns([1, 1])
    with col_meta1:
        st.markdown(f"<span style='font-size: 0.8rem; color: #64748b;'>{char_count:,} characters &bull; {word_count:,} words</span>", unsafe_allow_html=True)
    with col_meta2:
        if char_count > 100000:
            st.markdown("<span style='font-size: 0.8rem; color: #b91c1c; font-weight: 600;'>⚠️ Message exceeds 100KB limit (truncated for safety)</span>", unsafe_allow_html=True)

    # Action Buttons
    btn_col1, btn_col2, _ = st.columns([1.5, 1, 4])
    with btn_col1:
        analyze_clicked = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.input_email_text = ""
            st.rerun()

    # Inference Execution
    if analyze_clicked:
        if not email_text.strip():
            st.warning("⚠️ Please enter or paste an email message before analyzing.")
        else:
            # 1. Enforce max size limit (100KB)
            clean_input = email_text.strip()[:100000]

            # 2. Local Python Inference
            start_time = time.perf_counter()
            pred_class = pipeline.predict([clean_input])[0]
            pred_probs = pipeline.predict_proba([clean_input])[0]
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Pipeline classes: ['ham', 'spam']
            classes = list(pipeline.classes_)
            spam_idx = classes.index('spam') if 'spam' in classes else 1
            ham_idx = classes.index('ham') if 'ham' in classes else 0

            spam_prob = float(pred_probs[spam_idx])
            ham_prob = float(pred_probs[ham_idx])
            is_spam = (pred_class == 'spam')
            confidence = spam_prob if is_spam else ham_prob
            conf_pct = round(confidence * 100, 2)

            # 3. Store sanitized preview into SQLite database
            try:
                database.insert_prediction(
                    message=clean_input,
                    prediction=pred_class,
                    confidence=confidence
                )
            except Exception as e:
                st.error(f"Database logging error: {str(e)}")

            # 4. Display Formatted Result Card
            if is_spam:
                st.markdown(f"""
                <div class="threat-card-spam">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 1.8rem;">🚨</span>
                            <div>
                                <span class="badge-spam">SPAM / PHISHING DETECTED</span>
                                <h3 style="font-size: 1.25rem; font-weight: 800; color: #881337; margin-top: 4px; margin-bottom: 0;">
                                    Malicious or Unsolicited Communication
                                </h3>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #9f1239;">Model Confidence</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #881337;">{conf_pct}%</div>
                        </div>
                    </div>
                    <p style="font-size: 0.9rem; color: #4c0519; margin-bottom: 0; line-height: 1.5;">
                        This message exhibits word frequencies strongly correlated with unsolicited marketing, financial scams, or credential harvesting. Exercise caution and do not open links or download attachments.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="threat-card-ham">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 1.8rem;">✅</span>
                            <div>
                                <span class="badge-ham">VERIFIED LEGITIMATE (HAM)</span>
                                <h3 style="font-size: 1.25rem; font-weight: 800; color: #14532d; margin-top: 4px; margin-bottom: 0;">
                                    Normal Communication Profile
                                </h3>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #166534;">Model Confidence</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #14532d;">{conf_pct}%</div>
                        </div>
                    </div>
                    <p style="font-size: 0.9rem; color: #052e16; margin-bottom: 0; line-height: 1.5;">
                        The statistical distribution of terms in this message conforms closely to legitimate interpersonal and transactional business correspondence.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Probability Breakdown
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            prob_col1, prob_col2, prob_col3 = st.columns([1, 1, 1])
            with prob_col1:
                st.metric("Legitimate (Ham) Probability", f"{ham_prob * 100:.2f}%")
                st.progress(ham_prob)
            with prob_col2:
                st.metric("Spam Probability", f"{spam_prob * 100:.2f}%")
                st.progress(spam_prob)
            with prob_col3:
                st.metric("Inference Latency", f"{latency_ms:.2f} ms", delta="Local CPU")
                st.caption("Saved sanitized preview to SQLite audit history.")


# ------------------------------------------------------------------------------
# 5. VIEW: ANALYTICS DASHBOARD
# ------------------------------------------------------------------------------
elif page == "📊 Analytics Dashboard":
    st.markdown("""
    <div>
        <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background-color: #eef2ff; border: 1px solid #c7d2fe; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; color: #4338ca; margin-bottom: 10px;">
            📊 SQLite Telemetry
        </div>
        <h1 style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-bottom: 6px; tracking-tight;">
            Analytics & Threat Telemetry
        </h1>
        <p style="font-size: 0.95rem; color: #64748b; margin-bottom: 24px;">
            Real-time aggregate detection metrics computed strictly from local SQLite records.
        </p>
    </div>
    """, unsafe_allow_html=True)

    stats = database.get_stats()
    total = stats["total"]

    # 4 KPI Tiles
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-title">Total Analyzed</div>
            <div class="kpi-value">{total:,}</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Lifetime audit count</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-title">Spam Detected</div>
            <div class="kpi-value" style="color: #be123c;">{stats['spam']:,}</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Quarantined threats</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-title">Legitimate (Ham)</div>
            <div class="kpi-value" style="color: #15803d;">{stats['ham']:,}</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Clean communications</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-title">Spam Ratio</div>
            <div class="kpi-value" style="color: #4f46e5;">{stats['spam_percentage']}%</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Threat incident rate</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    if total == 0:
        st.info("ℹ️ No emails have been analyzed yet. Use the **Email Detection** tab to scan messages; real statistics and charts will appear here automatically.")
    else:
        chart_col1, chart_col2 = st.columns(2)

        # 1. Donut Chart (Altair)
        with chart_col1:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;'>Class Distribution</h3>", unsafe_allow_html=True)

            pie_df = pd.DataFrame([
                {"Category": "Legitimate (Ham)", "Count": stats["ham"], "Color": "#10b981"},
                {"Category": "Spam / Phishing", "Count": stats["spam"], "Color": "#f43f5e"},
            ])

            donut = alt.Chart(pie_df).mark_arc(innerRadius=60).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(
                    field="Category",
                    type="nominal",
                    scale=alt.Scale(domain=["Legitimate (Ham)", "Spam / Phishing"], range=["#10b981", "#f43f5e"]),
                    legend=alt.Legend(orient="bottom", title=None)
                ),
                tooltip=["Category", "Count"]
            ).properties(height=260)

            st.altair_chart(donut, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # 2. Activity Timeline Chart (Altair)
        with chart_col2:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;'>Detection Activity Timeline</h3>", unsafe_allow_html=True)

            activity = stats.get("activity_by_date", [])
            if activity:
                act_df = pd.DataFrame(activity)
                melted = pd.melt(act_df, id_vars=["date"], value_vars=["ham", "spam"], var_name="Type", value_name="Count")
                melted["Type"] = melted["Type"].map({"ham": "Legitimate (Ham)", "spam": "Spam Threats"})

                bar = alt.Chart(melted).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X("date:N", title="Date", axis=alt.Axis(labelAngle=-30)),
                    y=alt.Y("Count:Q", title="Analyses"),
                    color=alt.Color(
                        "Type:N",
                        scale=alt.Scale(domain=["Legitimate (Ham)", "Spam Threats"], range=["#10b981", "#f43f5e"]),
                        legend=alt.Legend(orient="bottom", title=None)
                    ),
                    tooltip=["date", "Type", "Count"]
                ).properties(height=260)

                st.altair_chart(bar, use_container_width=True)
            else:
                st.caption("Awaiting timeline history...")
            st.markdown("</div>", unsafe_allow_html=True)

        # Recent History Preview on Dashboard
        st.markdown("<h3 style='font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-top: 10px; margin-bottom: 12px;'>Recent Detections</h3>", unsafe_allow_html=True)
        recent_items, _ = database.get_history(limit=5)
        if recent_items:
            recent_df = pd.DataFrame(recent_items)
            recent_df["Confidence"] = recent_df["confidence"].apply(lambda c: f"{c*100:.2f}%")
            recent_df["Verdict"] = recent_df["prediction"].apply(lambda p: "🚨 SPAM" if p == "spam" else "✅ HAM")
            st.dataframe(
                recent_df[["id", "Verdict", "Confidence", "message_preview", "created_at"]].rename(
                    columns={"id": "ID", "message_preview": "Sanitized Preview", "created_at": "Timestamp"}
                ),
                use_container_width=True,
                hide_index=True
            )


# ------------------------------------------------------------------------------
# 6. VIEW: PREDICTION HISTORY (SQLITE AUDIT LOG)
# ------------------------------------------------------------------------------
elif page == "📜 Prediction History":
    st.markdown("""
    <div>
        <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background-color: #eef2ff; border: 1px solid #c7d2fe; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; color: #4338ca; margin-bottom: 10px;">
            🔒 Privacy-Preserving Audit Trail
        </div>
        <h1 style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-bottom: 6px; tracking-tight;">
            Prediction Audit History
        </h1>
        <p style="font-size: 0.95rem; color: #64748b; margin-bottom: 24px;">
            Search and filter past email predictions. For data privacy, only 140-character sanitized previews are stored.
        </p>
    </div>
    """, unsafe_allow_html=True)

    f_col1, f_col2, f_col3 = st.columns([3, 2, 1.5])
    with f_col1:
        search_query = st.text_input("🔍 Search Messages", placeholder="Search by preview text...")
    with f_col2:
        class_filter = st.selectbox("Filter by Verdict", ["All Verdicts", "Spam Only", "Ham Only"])
    with f_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        refresh_btn = st.button("🔄 Refresh", use_container_width=True)

    filter_val = None
    if class_filter == "Spam Only":
        filter_val = "spam"
    elif class_filter == "Ham Only":
        filter_val = "ham"

    items, total_count = database.get_history(
        limit=100,
        filter_type=filter_val,
        search_term=search_query
    )

    st.markdown(f"<p style='font-size: 0.85rem; color: #64748b; font-weight: 600;'>Showing {len(items)} of {total_count} matching records</p>", unsafe_allow_html=True)

    if items:
        hist_df = pd.DataFrame(items)
        hist_df["Status"] = hist_df["prediction"].apply(lambda p: "🚨 SPAM" if p == "spam" else "✅ HAM")
        hist_df["Confidence Score"] = hist_df["confidence"].apply(lambda c: f"{c*100:.2f}%")
        
        st.dataframe(
            hist_df[["id", "Status", "Confidence Score", "message_preview", "created_at"]].rename(
                columns={"id": "ID", "message_preview": "Sanitized Message Preview", "created_at": "Recorded At (UTC)"}
            ),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        # Clear History Section with Safety Confirmation
        with st.expander("🗑️ Clear Audit History"):
            st.warning("⚠️ This will permanently delete all stored prediction logs from SQLite. This action cannot be undone.")
            confirm_check = st.checkbox("I understand and wish to purge all prediction history")
            if st.button("Purge History Records", type="secondary", disabled=not confirm_check):
                database.clear_history()
                st.success("✅ Prediction history successfully purged.")
                st.rerun()
    else:
        st.info("ℹ️ No records match the current filter criteria.")


# ------------------------------------------------------------------------------
# 7. VIEW: MODEL ARCHITECTURE & EVALUATION
# ------------------------------------------------------------------------------
elif page == "🧠 Model Architecture":
    st.markdown("""
    <div>
        <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background-color: #eef2ff; border: 1px solid #c7d2fe; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; color: #4338ca; margin-bottom: 10px;">
            🔬 Scientific Machine Learning Methodology
        </div>
        <h1 style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-bottom: 6px; tracking-tight;">
            Model Architecture & Evaluation
        </h1>
        <p style="font-size: 0.95rem; color: #64748b; margin-bottom: 24px;">
            Every metric below reflects the exact holdout test evaluation from <code>Mail.ipynb</code> and <code>train_model.py</code> on unobserved test samples.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4 Highlights
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="kpi-tile">
            <div class="kpi-title">Test Accuracy</div>
            <div class="kpi-value" style="color: #10b981;">99.19%</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">1,106 of 1,115 samples correct</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="kpi-tile">
            <div class="kpi-title">Spam Precision</div>
            <div class="kpi-value" style="color: #4f46e5;">100.0%</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Zero false alarms (0% FP)</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="kpi-tile">
            <div class="kpi-title">Spam Recall</div>
            <div class="kpi-value" style="color: #d97706;">93.96%</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">140 of 149 test threats caught</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="kpi-tile">
            <div class="kpi-title">Vocabulary Size</div>
            <div class="kpi-value" style="color: #0f172a;">7,701</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Fitted CountVectorizer tokens</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Confusion Matrix & Classification Report Columns
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; color: #0f172a;'>Holdout Confusion Matrix (N = 1,115)</h3>", unsafe_allow_html=True)
        st.caption("Evaluated on 20% holdout test set with random_state=42")

        cm_table = pd.DataFrame(
            [
                ["Actual HAM (966)", "966 (True Negatives - 100%)", "0 (False Positives - 0%)"],
                ["Actual SPAM (149)", "9 (False Negatives - 6.04%)", "140 (True Positives - 93.96%)"]
            ],
            columns=["Ground Truth", "Predicted HAM", "Predicted SPAM"]
        )
        st.table(cm_table.set_index("Ground Truth"))
        st.info("💡 **Key Finding:** The model achieves a **0% False Positive Rate**, guaranteeing that genuine emails are never falsely quarantined as spam.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_col2:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; color: #0f172a;'>Classification Report Breakdown</h3>", unsafe_allow_html=True)
        st.caption("Standard Scikit-learn classification metrics across classes")

        cr_table = pd.DataFrame([
            {"Class": "Ham (Legitimate)", "Precision": "99.08%", "Recall": "100.00%", "F1-Score": "0.9954", "Support": 966},
            {"Class": "Spam (Malicious)", "Precision": "100.00%", "Recall": "93.96%", "F1-Score": "0.9689", "Support": 149},
            {"Class": "Macro Average", "Precision": "99.54%", "Recall": "96.98%", "F1-Score": "0.9821", "Support": 1115},
            {"Class": "Weighted Average", "Precision": "99.20%", "Recall": "99.19%", "F1-Score": "0.9918", "Support": 1115},
        ])
        st.table(cr_table.set_index("Class"))
        st.caption("F1-score demonstrates strong performance on the minority spam class (0.9689) despite natural class imbalance.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Dataset Profile & Feature Extraction
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown("""
        <div class="saas-card">
            <h3 style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">Dataset Profile (mail_data.csv)</h3>
            <ul style="font-size: 0.85rem; color: #334155; line-height: 1.7; padding-left: 20px;">
                <li><strong>Total Corpus Size:</strong> 5,572 labeled text samples.</li>
                <li><strong>Class Balance:</strong> 4,825 Ham (86.60%) vs. 747 Spam (13.40%).</li>
                <li><strong>Train/Test Ratio:</strong> 4,457 train / 1,115 holdout test.</li>
                <li><strong>Duplicate Messages:</strong> 415 duplicates detected in raw set.</li>
                <li><strong>Missing Data:</strong> 0 null fields across all records.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with d_col2:
        st.markdown("""
        <div class="saas-card">
            <h3 style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">Feature Extraction (CountVectorizer)</h3>
            <ul style="font-size: 0.85rem; color: #334155; line-height: 1.7; padding-left: 20px;">
                <li><strong>Word Tokenization:</strong> Extracts word n-grams with lowercasing.</li>
                <li><strong>Bag-of-Words:</strong> Builds token occurrence frequencies across 7,701 vocabulary items.</li>
                <li><strong>Unified Pipeline:</strong> Both vectorizer and MultinomialNB are bundled in a single Scikit-learn Pipeline object to strictly eliminate train-test leakage.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Mathematical Formulation
    st.markdown("""
    <div class="saas-card" style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #ffffff;">
        <h3 style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 6px;">Multinomial Naive Bayes Mathematical Theory</h3>
        <p style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
            Multinomial Naive Bayes estimates the probability that an incoming document with word count vector <code>x = (x₁, x₂, ..., xₙ)</code> belongs to class <code>c ∈ {ham, spam}</code>:
        </p>
        <div style="background-color: rgba(255,255,255,0.08); padding: 12px; border-radius: 8px; text-align: center; font-family: monospace; font-size: 1rem; color: #a5b4fc; margin: 12px 0;">
            P(c | x) ∝ P(c) · ∏ [ P(wᵢ | c) ]^(xᵢ)
        </div>
        <p style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6; margin-bottom: 0;">
            Where <code>P(wᵢ | c)</code> is estimated with Laplace smoothing (α = 1.0) to avoid zero-frequency anomalies when encountering unobserved vocabulary tokens.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Limitations
    with st.expander("⚠️ Known Model Limitations & Boundary Conditions"):
        st.markdown("""
        - **Bag-of-Words Limitation:** Word order and deep context nuances are discarded. Sarcasm or inverted phrases can be misclassified.
        - **Adversarial Typo Attacks:** Deliberately spaced or perturbed words (e.g., `f.r.e.e  c.a.s.h`) can evade exact token matching.
        - **Non-Calibrated Probabilities:** Extreme multiplication of feature likelihoods tends to push probabilities near 0.0 or 1.0. Confidence scores reflect model estimations rather than absolute mathematical certitude.
        """)


# ------------------------------------------------------------------------------
# 8. VIEW: ABOUT PROJECT & ARCHITECTURE
# ------------------------------------------------------------------------------
elif page == "ℹ️ About Project":
    st.markdown("""
    <div>
        <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background-color: #eef2ff; border: 1px solid #c7d2fe; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; color: #4338ca; margin-bottom: 10px;">
            🛡️ Enterprise Architecture
        </div>
        <h1 style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-bottom: 6px; tracking-tight;">
            About SpamGuard AI
        </h1>
        <p style="font-size: 0.95rem; color: #64748b; margin-bottom: 24px;">
            A privacy-first, standalone cybersecurity application designed to diagnose malicious and unsolicited electronic correspondence with zero third-party API exposure.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Problem & Solution
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("""
        <div class="saas-card">
            <h3 style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">The Problem</h3>
            <p style="font-size: 0.88rem; color: #475569; line-height: 1.6;">
                Unsolicited spam and deceptive phishing schemes account for significant global communication volume, leading to identity fraud, malware vectors, and wasted employee time. Simple keyword blocklists fail to generalize across syntactic permutations.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="saas-card">
            <h3 style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">The Solution</h3>
            <p style="font-size: 0.88rem; color: #475569; line-height: 1.6;">
                SpamGuard AI leverages probabilistic Natural Language Processing to classify messages in sub-millisecond execution. It runs fully offline on local Python CPU without sending private communications to external cloud LLMs.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Standalone Architecture Diagram
    st.markdown("""
    <div class="saas-card">
        <h3 style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;">Standalone Architecture (No REST API / No Microservices)</h3>
        <div style="background-color: #f1f5f9; padding: 18px; border-radius: 10px; font-family: monospace; font-size: 0.82rem; color: #1e293b; line-height: 1.7; overflow-x: auto;">
            [ User Browser Interface ]
                    │
                    ▼  (Direct Python Execution via Streamlit)
            ┌────────────────────────────────────────────────────────┐
            │                 Streamlit Web App                      │
            │                  (app.py - Python)                     │
            └───────────────┬────────────────────────┬───────────────┘
                            │                        │
                            ▼                        ▼
            ┌──────────────────────────────┐  ┌──────────────────────┐
            │  Scikit-Learn Pipeline (.pkl)│  │   SQLite Database    │
            │  CountVectorizer + MNB Model │  │   (spam_guard.db)    │
            │  (Sub-millisecond inference) │  │  (Sanitized previews)│
            └──────────────────────────────┘  └──────────────────────┘
        </div>
        <p style="font-size: 0.82rem; color: #64748b; margin-top: 10px; margin-bottom: 0;">
            <strong>Zero Network Overhead:</strong> Eliminates HTTP overhead, port collisions, cross-origin resource sharing (CORS) errors, and server synchronization issues.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Tech Stack
    st.markdown("""
    <div class="saas-card">
        <h3 style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;">Technology Stack</h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
            <tr style="border-bottom: 1px solid #e2e8f0; text-align: left;">
                <th style="padding: 8px 0; color: #64748b;">Layer</th>
                <th style="padding: 8px 0; color: #64748b;">Technology</th>
                <th style="padding: 8px 0; color: #64748b;">Role</th>
            </tr>
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 10px 0; font-weight: 600; color: #0f172a;">Application Framework</td>
                <td style="padding: 10px 0;">Streamlit (Python)</td>
                <td style="padding: 10px 0; color: #475569;">Interactive user interface, layout, state management</td>
            </tr>
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 10px 0; font-weight: 600; color: #0f172a;">Machine Learning</td>
                <td style="padding: 10px 0;">Scikit-learn (MultinomialNB)</td>
                <td style="padding: 10px 0; color: #475569;">CountVectorizer + Naive Bayes text classification pipeline</td>
            </tr>
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 10px 0; font-weight: 600; color: #0f172a;">Model Serialization</td>
                <td style="padding: 10px 0;">Joblib</td>
                <td style="padding: 10px 0; color: #475569;">Fast disk loading and in-memory model persistence</td>
            </tr>
            <tr style="border-bottom: 1px solid #f1f5f9;">
                <td style="padding: 10px 0; font-weight: 600; color: #0f172a;">Embedded Database</td>
                <td style="padding: 10px 0;">SQLite3</td>
                <td style="padding: 10px 0; color: #475569;">Zero-configuration persistence for audit logs and metrics</td>
            </tr>
            <tr>
                <td style="padding: 10px 0; font-weight: 600; color: #0f172a;">Data Visualization</td>
                <td style="padding: 10px 0;">Altair & Pandas</td>
                <td style="padding: 10px 0; color: #475569;">Interactive SVG charts for class distribution & timelines</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Privacy & Disclaimer
    st.markdown("""
    <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 12px; padding: 18px; font-size: 0.82rem; color: #475569; line-height: 1.6;">
        <strong>🛡️ Privacy & Operational Disclaimer:</strong><br>
        SpamGuard AI does not store full email contents. Only 140-character sanitized previews are logged to SQLite for audit and analytics.
        Predictions are probabilistic estimates derived from statistical text modeling. Users should exercise caution when dealing with sensitive communications.
    </div>
    """, unsafe_allow_html=True)
