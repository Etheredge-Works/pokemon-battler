from sklearn.preprocessing import LabelEncoder
import pandas as pd


def create_encoder(file_path: str, label: str) -> dict:
    df = pd.read_csv(file_path)
    items = df[label].values
    le = LabelEncoder()
    le.fit(items)
    return le
