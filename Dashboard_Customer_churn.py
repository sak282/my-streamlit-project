
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Insight AI",
    page_icon="📊",
    layout="wide"
)

# ---------- CUSTOM DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef4ff 0%, #f8faff 100%);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1150px;
}
.hero {
    background: linear-gradient(120deg, #102a43, #176b87);
    padding: 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(16, 42, 67, 0.15);
}
.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
}
.hero p {
    color: #d9f0ff;
    font-size: 17px;
}
.section-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #dce7f5;
    box-shadow: 0 5px 18px rgba(25, 55, 90, 0.06);
}
div.stButton > button {
    background: linear-gradient(90deg, #176b87, #2589a8);
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-weight: 600;
    width: 100%;
}
div.stButton > button:hover {
    background: #102a43;
    color: white;
    border: none;
}
div[data-testid="stNumberInput"] input {
    background-color: #f5f8fc;
}
.footer {
    text-align: center;
    color: #62748a;
    padding: 20px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return joblib.load("Marvellous_ML_Customer_Churn.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model. Please check that the model file is uploaded.")
    st.stop()

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <h1>📊 Customer Insight AI</h1>
    <p>Customer Churn Prediction Dashboard</p>
    <p>Use machine learning to identify customers who may leave your service.</p>
</div>
""", unsafe_allow_html=True)

# ---------- SUMMARY CARDS ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.info("📈 **Smart Analytics**")

with col2:
    st.info("🤖 **Machine Learning**")

with col3:
    st.info("🎯 **Churn Prediction**")

st.write("")

# ---------- CUSTOMER INPUT ----------
st.markdown("""
<div class="section-card">
    <h3 style="color:#102a43;">👤 Customer Details</h3>
    <p style="color:#62748a;">
        Enter the customer's information below to predict churn risk.
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "🎂 Customer Age",
            min_value=1,
            max_value=120,
            value=30,
            step=1
        )

        tenure = st.number_input(
            "📅 Tenure (months)",
            min_value=0,
            value=12,
            step=1
        )

    with col2:
        charges = st.number_input(
            "💳 Monthly Charges",
            min_value=0.0,
            value=50.0,
            step=5.0
        )

        calls = st.number_input(
            "☎️ Support Calls",
            min_value=0,
            value=1,
            step=1
        )

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn"
    )

# ---------- PREDICTION ----------
if submitted:
    new_customer = pd.DataFrame({
        "Age": [age],
        "Tenure": [tenure],
        "MonthlyCharges": [charges],
        "SupportCalls": [calls]
    })

    try:
        prediction = int(model.predict(new_customer)[0])

        st.markdown("---")
        st.subheader("📋 Prediction Result")

        if prediction == 1:
            st.error(
                "⚠️ **Customer is likely to churn**\n\n"
                "The model predicts that this customer may leave the service."
            )
        else:
            st.success(
                "✅ **Customer is unlikely to churn**\n\n"
                "The model predicts that this customer is likely to stay."
            )

        with st.expander("View customer details"):
            st.dataframe(new_customer, use_container_width=True)

    except Exception:
        st.error(
            "Prediction failed. Please check that the input columns "
            "match the features used when training the model."
        )

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    Customer Insight AI · Machine Learning Dashboard<br>
    Customer Churn Prediction System
</div>
""", unsafe_allow_html=True)
