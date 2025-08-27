# 🛒 Cartlytics

An interactive Streamlit app for Market Basket Analysis using the FP-Growth algorithm.
Upload a CSV of transactions, tune support/confidence/lift thresholds, and explore:

- 📦 Top items purchased

- 🧩 Frequent itemsets

- 🔗 Association rules (with metrics: support, confidence, lift, leverage, conviction)

- 📊 Visuals: bar charts, scatter plots, and a rule network graph

- 💾 Download itemsets and rules as CSV

## 📂 Project Structure
```
cartlytics-main/
├─ app.py              # Main Streamlit app
├─ utils.py            # Helper functions (CSV loading, encoding, top items)
├─ requirements.txt    # Python dependencies
├─ README.md           # This file
└─ .venv/              # (Optional) Virtual environment
```
## ⚙️ Requirements
Python 3.9+, pip (latest recommended), Dependencies

See requirements.txt. 
```
streamlit==1.36.0
pandas>=2.0
numpy>=1.23
mlxtend==0.23.1            # FP-Growth, association rules
scikit-learn>=1.3
plotly>=5.19               # charts
pyvis==0.3.2               # network graph
matplotlib>=3.7
jinja2>=3.1                #required for pyvis on Windows
```

## 🚀 Setup & Run
1. Clone / copy the project
```
git clone <your-repo-url>
cd cartlytics-main
```
2. Create & activate a virtual environment
```
# Windows (PowerShell)

python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Run the Streamlit app
```
streamlit run app.py
```

Streamlit will show URLs like:
```
Local URL: http://localhost:8501
Network URL: http://192.168.xx.xx:8501
```

Open the Local URL in your browser.

## 📑 How to Use

1. Upload a CSV:

- Each row = a basket/transaction

- Columns = items (Item_1, Item_2, ...)
- Example: Market_Basket_Optimisation.csv

2. Adjust parameters in the sidebar:

- min_support (fraction of transactions)

- min_confidence (rule strength)

- min_lift (filter for meaningful rules)

3. Explore:

- Top items bar chart

- Frequent itemsets bar chart & table

- Association rules scatter plot & table

- Interactive network graph of rules

4. Download:

- Itemsets CSV

- Rules CSV

## 🧪 Example Dataset

The [Market Basket Optimisation dataset (7501 transactions)](https://www.kaggle.com/datasets/andrewtoh78/market-basket-optimisation)
 works out-of-the-box.

## 🛠 Troubleshooting

- If you see frozenset errors → use the provided updated app.py and utils.py (they stringify itemsets).

- If PyVis throws template errors → ensure jinja2 is installed:
```
pip install jinja2
```
## ✨ Screenshots

- Top Items Frequency (bar chart)

- Frequent Itemsets (bar + table)

- Association Rules (scatter, table)

- Rule Network (interactive)

# 📜 License

MIT License – free to use, modify, and distribute.
