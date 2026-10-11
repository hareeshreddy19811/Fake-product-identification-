# Fake Product Identification System

A Streamlit web app that estimates whether a product is likely genuine or
fake, using a Random Forest classifier and three simple authenticity checks.

> **Educational demo only.** The model is trained on a tiny synthetic dataset
> and does not verify real products. Always confirm serial numbers with the
> actual manufacturer.

## What it does

- Collects product name, brand name and serial number
- Asks three yes/no authenticity questions:
  - Is the brand verified against a trusted source?
  - Is the serial number valid in the manufacturer's records?
  - Does the price match the manufacturer's or authorized seller's listing?
- Runs a `RandomForestClassifier` (scikit-learn) on those answers
- Shows a "Likely Genuine" / "Potentially Fake" verdict with a genuine score

## Project structure

```
fake-product-identification/
├── app.py              # Streamlit app
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/fake-product-identification.git
cd fake-product-identification
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501), fill in
the product details, and click **Identify Product**.

## Limitations

- Training data is a hand-made 20-row table, so scores are illustrative only
- The three checks are answered by the user; the app does not look up brands,
  serial numbers or prices itself

## License

MIT
