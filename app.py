"""
Fake Product Identification System (Streamlit app)
---------------------------------------------------
Educational demo that uses a Random Forest classifier, trained on a tiny
synthetic dataset, to estimate whether a product is likely genuine based on
three checks: brand verification, serial number validity, and price match.

Run:
    streamlit run app.py

NOTE: This is a demonstration only. It does not verify real products.
"""

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["brand_verified", "serial_valid", "price_match"]

# Sample training dataset for demonstration only
TRAINING_DATA = pd.DataFrame(
    {
        "brand_verified": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        "serial_valid": [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
        "price_match": [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
        "authentic": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    }
)


@st.cache_resource
def train_model() -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(TRAINING_DATA[FEATURES], TRAINING_DATA["authentic"])
    return model


def predict(model: RandomForestClassifier, brand_ok: bool, serial_ok: bool, price_ok: bool):
    features = pd.DataFrame(
        [
            {
                "brand_verified": int(brand_ok),
                "serial_valid": int(serial_ok),
                "price_match": int(price_ok),
            }
        ]
    )
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    genuine_score = probabilities[list(model.classes_).index(1)]
    return prediction, genuine_score


def main():
    st.set_page_config(page_title="Fake Product Identification", page_icon="🔍")

    st.title("🔍 Fake Product Identification System")
    st.write("Enter product details to check its authenticity.")

    model = train_model()

    st.subheader("Enter Product Information")

    product_name = st.text_input("Product Name")
    brand_name = st.text_input("Brand Name")
    brand_verified = st.selectbox(
        "Is the brand verified against a trusted source?", ["Yes", "No"]
    )
    serial_number = st.text_input("Product Serial Number")
    serial_valid = st.selectbox(
        "Is the serial number valid in the manufacturer's records?", ["Yes", "No"]
    )
    price_match = st.selectbox(
        "Does the price match the manufacturer's or authorized seller's listing?",
        ["Yes", "No"],
    )

    if st.button("Identify Product"):
        if not product_name.strip() or not brand_name.strip() or not serial_number.strip():
            st.warning("Please enter the product name, brand name, and serial number.")
            return

        prediction, score = predict(
            model,
            brand_verified == "Yes",
            serial_valid == "Yes",
            price_match == "Yes",
        )

        st.subheader("Product Verification Result")
        st.write("Product:", product_name)
        st.write("Brand:", brand_name)
        st.write("Serial Number:", serial_number)

        if prediction == 1:
            st.success("Result: Likely Genuine")
        else:
            st.error("Result: Potentially Fake")

        st.metric("Model's Genuine Score", f"{score:.1%}")

        st.caption(
            "Educational demonstration only. The model uses a tiny "
            "synthetic dataset and does not verify real products. "
            "Check serial numbers with the actual manufacturer."
        )


if __name__ == "__main__":
    main()
