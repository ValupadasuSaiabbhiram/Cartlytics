import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pyvis.network import Network

from mlxtend.frequent_patterns import fpgrowth, association_rules

from utils import load_transactions_from_csv, to_one_hot_dataframe, top_items_dataframe

st.set_page_config(
    page_title="Cartlytics",
    page_icon="🛒",
    layout="wide",
)

# --- Title & intro ---
st.title("🛒 Cartlytics - FP-Growth Market Basket Dashboard")
st.caption("Upload a transactions CSV, tune thresholds, and explore frequent itemsets & rules. Download CSVs, too!")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader(
        "Upload CSV (rows = baskets; columns = Item_1, Item_2, ...)",
        type=["csv"],
        help="You can upload the 'Market_Basket_Optimisation.csv' or any similar file.",
    )
    path_input = st.text_input(
        "…or provide a CSV path",
        value="",
        placeholder="C:/path/to/Market_Basket_Optimisation.csv",
    )
    has_header = st.selectbox("Header row in CSV?", ["Auto-detect / Unknown", "Yes", "No"], index=0)
    hdr = None if has_header == "Auto-detect / Unknown" else (has_header == "Yes")

    st.divider()
    st.subheader("FP-Growth Parameters")
    min_support = st.slider("Minimum Support", 0.001, 0.2, 0.04, 0.001, help="Proportion of transactions")
    min_conf = st.slider("Minimum Confidence", 0.05, 1.0, 0.30, 0.01)
    min_lift = st.slider("Minimum Lift (filter on rules)", 0.5, 5.0, 1.0, 0.05)

    st.divider()
    top_n_items = st.slider("Show Top N Items", 5, 60, 30, 1)
    max_rules_for_graph = st.slider("Network: Max rules", 10, 200, 60, 5)

# --- Load data ---
@st.cache_data(show_spinner=True)
def _load_data(uploaded, path_input, hdr):
    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        df_raw, txns = load_transactions_from_csv(raw_bytes, has_header=hdr)
    elif path_input.strip():
        df_raw, txns = load_transactions_from_csv(path_input.strip(), has_header=hdr)
    else:
        st.stop()
    return df_raw, txns

if not uploaded and not path_input.strip():
    st.info("Upload a CSV or enter a local CSV path in the sidebar to begin.")
    st.stop()

df_raw, transactions = _load_data(uploaded, path_input, hdr)
n_tx = len(transactions)
unique_items = len({x for row in transactions for x in row})

# --- KPI Row ---
col1, col2, col3 = st.columns(3)
col1.metric("Transactions", f"{n_tx:,}")
col2.metric("Unique Items", f"{unique_items:,}")
col3.metric("Min Support / Confidence", f"{min_support:.3f} / {min_conf:.2f}")

st.divider()

# --- Top Items ---
st.subheader("📦 Item Frequency")
top_df = top_items_dataframe(transactions)
top_df["% Count"] = (100 * top_df["Count"] / n_tx).round(2)

c1, c2 = st.columns([2, 1])
with c1:
    top_show = top_df.head(top_n_items)
    fig = px.bar(top_show, x="Item", y="Count", title=f"Top {len(top_show)} Items", text="Count")
    fig.update_layout(xaxis_tickangle=-45, height=420)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.dataframe(top_show, use_container_width=True, hide_index=True)

# --- One-hot encode ---
@st.cache_data(show_spinner=False)
def _one_hot(transactions):
    return to_one_hot_dataframe(transactions)

df_onehot = _one_hot(transactions)

# --- FP-Growth ---
@st.cache_data(show_spinner=True)
def _mine_itemsets_and_rules(df_onehot, min_support, min_conf):
    itemsets = fpgrowth(df_onehot, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        return itemsets, pd.DataFrame()
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_conf)
    rules = rules.sort_values(["confidence", "support", "lift"], ascending=[False, False, False])
    return itemsets, rules

frequent_itemsets, rules = _mine_itemsets_and_rules(df_onehot, min_support, min_conf)

st.divider()
st.subheader("🧩 Frequent Itemsets")

if frequent_itemsets.empty:
    st.warning("No frequent itemsets found. Try lowering the minimum support.")
else:
    # Convert frozenset -> string; DO NOT show the raw frozenset column
    fi = frequent_itemsets.copy()
    fi["itemset"] = fi["itemsets"].apply(lambda x: ", ".join(sorted(list(x))))
    fi["support %"] = (100 * fi["support"]).round(2)

    # ---- Bar chart (top 30) ----
    top_fi = fi.sort_values("support", ascending=False).head(30)
    fig2 = px.bar(
        top_fi,
        x="itemset",
        y="support %",
        title="Top Frequent Itemsets (by support)",
        text="support %"
    )
    fig2.update_layout(xaxis_tickangle=-45, height=420)
    st.plotly_chart(fig2, use_container_width=True)

    # ---- Table: drop the raw 'itemsets' frozenset column ----
    fi_display = fi[["itemset", "support", "support %"]].sort_values("support", ascending=False)
    st.dataframe(fi_display, use_container_width=True, hide_index=True)

    # ---- Download ----
    csv_fi = fi_display.to_csv(index=False).encode("utf-8")
    st.download_button("Download Frequent Itemsets (CSV)", csv_fi, "frequent_itemsets.csv", "text/csv")

st.divider()
st.subheader("🔗 Association Rules")

if rules.empty:
    st.info("No rules generated. Try lowering min support/confidence.")
else:
    # Stringify sets for display
    rules_disp = rules.copy()
    rules_disp["antecedents"] = rules_disp["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules_disp["consequents"] = rules_disp["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    # Filter by lift
    rules_disp = rules_disp[rules_disp["lift"] >= min_lift]

    pretty = rules_disp.rename(
        columns={
            "antecedents":"Antecedent",
            "consequents":"Consequent",
            "support":"Support",
            "confidence":"Confidence",
            "lift":"Lift",
            "leverage":"Leverage",
            "conviction":"Conviction"
        }
    )
    pretty["Support %"] = (100 * pretty["Support"]).round(2)
    pretty["Confidence %"] = (100 * pretty["Confidence"]).round(2)
    pretty["Lift"] = pretty["Lift"].round(3)

    # Scatter: Support vs Confidence
    fig3 = px.scatter(
        pretty,
        x="Support",
        y="Confidence",
        size="Confidence",
        color="Lift",
        hover_data=["Antecedent","Consequent","Support %","Confidence %","Lift"],
        title="Support vs Confidence (colored by Lift)"
    )
    fig3.update_layout(height=520)
    st.plotly_chart(fig3, use_container_width=True)

    # Table
    st.dataframe(
        pretty[["Antecedent","Consequent","Support %","Confidence %","Lift","Leverage","Conviction"]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

    # Downloads
    csv_rules = pretty.to_csv(index=False).encode("utf-8")
    st.download_button("Download Rules (CSV)", csv_rules, "association_rules.csv", "text/csv")

    st.markdown("### 🕸️ Rule Network (top rules)")
    subset = pretty.sort_values(["Lift","Confidence","Support"], ascending=False).head(max_rules_for_graph)

    # PyVis network (use write_html(..., notebook=False) to avoid Jinja template bug on Windows)
    net = Network(height="560px", width="100%", bgcolor="#FFFFFF", font_color="#0F172A", notebook=False, directed=True)
    net.barnes_hut(gravity=-4000, central_gravity=0.3, spring_length=110, spring_strength=0.03)

    # Add nodes & edges
    def add_node_if_missing(label, color="#E2E8F0"):
        if label not in [n["id"] for n in net.nodes]:
            net.add_node(label, label=label, color=color)

    for _, row in subset.iterrows():
        a = row["Antecedent"]
        c = row["Consequent"]
        lift = float(row["Lift"])
        conf = float(row["Confidence %"])
        sup = float(row["Support %"])

        add_node_if_missing(a, color="#DBEAFE")
        add_node_if_missing(c, color="#FDE68A")
        net.add_edge(a, c, value=lift, title=f"lift={lift}, conf={conf}%, support={sup}%")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.write_html(tmp.name, notebook=False)
        html_code = Path(tmp.name).read_text(encoding="utf-8")
        st.components.v1.html(html_code, height=580, scrolling=True)
