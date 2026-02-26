import pandas as pd


def cargar_datos(path: str):
    return pd.read_csv(path)
