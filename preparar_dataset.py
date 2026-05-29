import pandas as pd
import os

data_dir = r"C:\Users\ma_va\Documents\Salary\DATA"

# 1. Definir solo las columnas confirmadas
cols_llaves = ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN']
cols_carac = cols_llaves + ['P6040', 'P3271', 'P3042', 'P6090']
cols_ocup = cols_llaves + ['RAMA2D_R4', 'P6430', 'P6426', 'P6800', 'P6880', 'INGLABO']

print("📥 Cargando datos en memoria (solo columnas útiles)...")
df_carac = pd.read_csv(os.path.join(data_dir, "caracteristicas_generales_2025.csv"), sep=';', usecols=cols_carac)
df_ocup = pd.read_csv(os.path.join(data_dir, "ocupados_2025.csv"), sep=';', usecols=cols_ocup)

print("🔀 Cruzando tablas...")
df_master = pd.merge(df_carac, df_ocup, on=cols_llaves, how='inner')

print("🧹 Limpiando datos y renombrando variables...")
df_master.rename(columns={
    'P6040': 'edad',
    'P3271': 'sexo',
    'P3042': 'nivel_educativo',
    'P6090': 'afiliado_salud',
    'RAMA2D_R4': 'sector_economico',
    'P6430': 'tipo_contrato',
    'P6426': 'meses_experiencia',
    'P6800': 'horas_semanales',
    'P6880': 'tamano_empresa',
    'INGLABO': 'salario_mensual'
}, inplace=True)

# Eliminar nulos en el Target y en features esenciales
df_master.dropna(subset=['salario_mensual', 'nivel_educativo', 'sector_economico'], inplace=True)

# Filtro de cordura: Salarios mayores a 500k COP (para no perder a los de medio tiempo)
df_master = df_master[df_master['salario_mensual'] >= 500000] 

# Eliminar llaves, ya no las necesitamos
df_master.drop(columns=cols_llaves, inplace=True)

output_path = os.path.join(data_dir, "dataset_modelo_limpio.csv")
df_master.to_csv(output_path, index=False)

print(f"✅ ¡Dataset final listo! Guardado en: {output_path}")
print(f"📊 Total de registros útiles: {len(df_master):,}")