import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import linear_kernel


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AmazonHelp AI Support Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(255, 153, 0, 0.08), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(88, 101, 242, 0.08), transparent 30%),
            #0b0d12;
        color: #f5f7fa;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: #10131a;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] h2 {
        color: #ffffff;
    }

    /* ---------- HERO ---------- */
    .hero {
        padding: 28px 32px;
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                rgba(255,153,0,0.15),
                rgba(88,101,242,0.10)
            );
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 24px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.25);
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 8px;
        color: #ffffff;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #aeb6c5;
        line-height: 1.6;
        max-width: 850px;
    }

    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,153,0,0.12);
        color: #ffad33;
        border: 1px solid rgba(255,153,0,0.25);
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* ---------- INPUT ---------- */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 24px 0 10px 0;
        color: #ffffff;
    }

    div[data-testid="stTextArea"] textarea {
        background: #161a23 !important;
        color: #f5f7fa !important;
        border: 1px solid #2b3240 !important;
        border-radius: 14px !important;
        padding: 16px !important;
        font-size: 1rem !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border: 1px solid #ff9900 !important;
        box-shadow: 0 0 0 1px rgba(255,153,0,0.25) !important;
    }

    /* ---------- BUTTON ---------- */
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 48px;
        border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(255,153,0,0.45);
        box-shadow: 0 8px 24px rgba(255,153,0,0.12);
    }

    /* ---------- RESULT CARDS ---------- */
    .card {
        background: #131720;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 20px;
        margin-top: 12px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.18);
    }

    .card-label {
        color: #8f99aa;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }

    .card-value {
        color: #ffffff;
        font-size: 1.18rem;
        font-weight: 700;
    }

    .metric-number {
        color: #ffffff;
        font-size: 1.55rem;
        font-weight: 800;
    }

    /* ---------- DECISION ---------- */
    .decision {
        padding: 18px 20px;
        border-radius: 16px;
        margin: 18px 0;
        font-weight: 800;
        font-size: 1.15rem;
        text-align: center;
    }

    .auto {
        background: rgba(34,197,94,0.10);
        border: 1px solid rgba(34,197,94,0.30);
        color: #4ade80;
    }

    .escalate {
        background: rgba(239,68,68,0.10);
        border: 1px solid rgba(239,68,68,0.30);
        color: #f87171;
    }

    /* ---------- RESPONSE ---------- */
    .response-box {
        background: #171b25;
        border: 1px solid #2b3240;
        border-left: 4px solid #ff9900;
        border-radius: 14px;
        padding: 20px;
        color: #e8ebf0;
        line-height: 1.7;
        margin-top: 10px;
    }

    /* ---------- INFO ---------- */
    .info-box {
        background: rgba(88,101,242,0.07);
        border: 1px solid rgba(88,101,242,0.18);
        border-radius: 14px;
        padding: 16px;
        color: #b9c2d0;
        line-height: 1.6;
        font-size: 0.9rem;
    }

    /* ---------- FOOTER ---------- */
    .footer {
        margin-top: 35px;
        text-align: center;
        color: #6f7888;
        font-size: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_resource
def load_agent():

    train = pd.read_csv("amazonhelp_weak_training.csv")
    pairs = pd.read_csv("amazonhelp_response_pairs.csv")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(
        train["clean_text"].fillna(""),
        train["weak_intent"],
    )

    retrieval_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
    )

    retrieval_matrix = retrieval_vectorizer.fit_transform(
        pairs["clean_text"].fillna("")
    )

    return model, pairs, retrieval_vectorizer, retrieval_matrix


model, pairs, retrieval_vectorizer, retrieval_matrix = load_agent()


# =========================================================
# AGENT
# =========================================================
def support_agent(query):

    probabilities = model.predict_proba([query])[0]
    classes = model.named_steps["clf"].classes_

    best_index = probabilities.argmax()

    intent = classes[best_index]
    confidence = float(probabilities[best_index])

    query_vector = retrieval_vectorizer.transform([query])

    scores = linear_kernel(
        query_vector,
        retrieval_matrix,
    ).ravel()

    best_response_index = scores.argmax()

    response = pairs.iloc[best_response_index]["support_reply"]

    similarity = float(scores[best_response_index])

    high_risk_intents = {
        "account_payment_security",
        "seller_marketplace",
    }

    if (
        confidence < 0.60
        or similarity < 0.25
        or intent in high_risk_intents
    ):
        decision = "ESCALATE"
    else:
        decision = "AUTO-HANDLE"

    return {
        "intent": intent,
        "confidence": confidence,
        "similarity": similarity,
        "response": response,
        "decision": decision,
    }


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## 🤖 AmazonHelp AI")

    st.markdown(
        """
        <div class="info-box">
        <b>Pipeline</b><br><br>
        1. Intent classification<br>
        2. Historical response retrieval<br>
        3. Confidence assessment<br>
        4. AUTO-HANDLE / ESCALATE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Example queries")

    samples = [
        "My package says delivered but I never received it.",
        "Someone hacked my Amazon account.",
        "I want to return my damaged product.",
        "Where is my package? It was due yesterday.",
    ]

    selected_sample = None

    for sample in samples:
        if st.button(sample, use_container_width=True):
            selected_sample = sample

    st.markdown("---")

    st.caption(
        "Responses are retrieved from historical AmazonHelp "
        "support interactions."
    )


# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="badge">AI CUSTOMER SUPPORT • AMAZONHELP</div>
        <div class="hero-title">
            Support Intelligence Console
        </div>
        <div class="hero-subtitle">
            Classify customer intent, retrieve a historically grounded
            support response, and decide whether the request should be
            automatically handled or escalated to a human agent.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# INPUT
# =========================================================
st.markdown(
    '<div class="section-title">Customer message</div>',
    unsafe_allow_html=True,
)

default_text = selected_sample if "selected_sample" in locals() and selected_sample else ""

query = st.text_area(
    "",
    value=default_text,
    height=150,
    placeholder="Example: My package says delivered but I never received it.",
    label_visibility="collapsed",
)


analyze = st.button(
    "⚡ Analyze Customer Message",
    type="primary",
    use_container_width=True,
)

# =========================================================
# ANALYSIS
# =========================================================
if analyze:

    if not query.strip():
        st.warning("Please enter a customer message.")

    else:

        with st.spinner("Analyzing customer request..."):
            result = support_agent(query.strip())

        st.markdown("### Analysis")

        # -------------------------------------------------
        # Customer message being analyzed
        # -------------------------------------------------
        st.markdown(
            """
            <div class="section-title">
                Customer message analyzed
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="response-box"
                 style="
                    border-left: 4px solid #5865f2;
                    background: #151923;
                 ">
                {query.strip()}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # Decision banner
        # -------------------------------------------------
        if result["decision"] == "AUTO-HANDLE":
            st.markdown(
                """
                <div class="decision auto">
                    🟢 AUTO-HANDLE — Safe to answer automatically
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                """
                <div class="decision escalate">
                    🔴 ESCALATE — Human review recommended
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-label">Predicted Intent</div>
                    <div class="card-value">
                        {result["intent"].replace("_", " ").title()}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-label">Model Confidence</div>
                    <div class="metric-number">
                        {result["confidence"] * 100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-label">Response Match Score</div>
                    <div class="metric-number">
                        {result["similarity"]:.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------
        # Why this decision?
        # -------------------------------------------------
        confidence = result["confidence"]
        similarity = result["similarity"]
        intent = result["intent"]

        reasons = []

        if confidence < 0.60:
            reasons.append(
                f"Model confidence is {confidence * 100:.1f}%, "
                "below the 60% AUTO-HANDLE threshold."
            )

        if similarity < 0.25:
            reasons.append(
                f"Response match score is {similarity:.3f}, "
                "below the 0.25 grounding threshold."
            )

        if intent in {
            "account_payment_security",
            "seller_marketplace",
        }:
            reasons.append(
                f"'{intent.replace('_', ' ').title()}' is treated "
                "as a higher-risk intent and requires human review."
            )

        if not reasons:
            reasons.append(
                "Confidence, response-match score, and intent risk "
                "all passed the AUTO-HANDLE checks."
            )

        st.markdown("### Why this decision?")

        st.markdown(
            f"""
            <div class="info-box">
                {"<br><br>".join("• " + r for r in reasons)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # Suggested response
        # -------------------------------------------------
        st.markdown("### Suggested Response")

        st.markdown(
            f"""
            <div class="response-box">
                {result["response"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "This response is retrieved from historical AmazonHelp "
            "customer-support interactions rather than freely generated."
        )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        AmazonHelp AI Customer Support Agent • Intent + Retrieval + Triage
    </div>
    """,
    unsafe_allow_html=True,
)