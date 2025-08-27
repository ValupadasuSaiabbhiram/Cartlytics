import io
from typing import List, Tuple
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder

def load_transactions_from_csv(
    file: str | bytes,
    has_header: bool | None = None,
) -> Tuple[pd.DataFrame, List[List[str]]]:
    """
    Accepts:
      - uploaded file-like object (bytes) OR a filepath string.
    Returns:
      - raw DataFrame (as read)
      - list of transactions: List[List[str]]
    """
    if isinstance(file, (bytes, bytearray)):
        df = pd.read_csv(
            io.BytesIO(file),
            header=0 if has_header else None,
            on_bad_lines="skip",
            encoding_errors="ignore",
        )
    else:
        df = pd.read_csv(
            file,
            header=0 if has_header else None,
            on_bad_lines="skip",
            encoding_errors="ignore",
        )

    # If no header provided, give generic names
    if df.columns.dtype == "int64" or any(str(c).isdigit() for c in df.columns):
        df.columns = [f"Item_{i}" for i in range(1, df.shape[1] + 1)]

    # Convert rows to cleaned transactions
    values = df.fillna("").astype(str).values.tolist()
    txns = [[x.strip() for x in row if x and x.strip() and x.lower() != "nan"] for row in values]
    txns = [row for row in txns if len(row) > 0]
    return df, txns

def to_one_hot_dataframe(transactions: List[List[str]]) -> pd.DataFrame:
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_)

def top_items_dataframe(transactions: List[List[str]]) -> pd.DataFrame:
    items = pd.Series([item for row in transactions for item in row])
    vc = items.value_counts().rename_axis("Item").reset_index(name="Count")
    return vc
