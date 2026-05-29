import pandas as pd
import os

data_dir = r"C:\Users\ma_va\Documents\Salary\DATA"

print("📊 Variables en Características Generales:")
# Leemos solo la fila 0 (encabezados)
df_carac = pd.read_csv(os.path.join(data_dir, "caracteristicas_generales_2025.csv"), sep=';', nrows=0)
print(list(df_carac.columns))

print("\n💼 Variables en Ocupados:")
df_ocup = pd.read_csv(os.path.join(data_dir, "ocupados_2025.csv"), sep=';', nrows=0)
print(list(df_ocup.columns))