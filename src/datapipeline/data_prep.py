import pandas as pd
import numpy as np


def preparar_datos_dane(ruta_caracteristicas, ruta_ocupados):
    # 1. Cargar los CSVs (Asumiendo que los descargas y extraes)
    print("Cargando microdatos del DANE...")
    df_carac = pd.read_csv(ruta_caracteristicas, sep=';', low_memory=False)
    df_ocupados = pd.read_csv(ruta_ocupados, sep=';', low_memory=False)

    # 2. Cruce de tablas (Merge)
    # DIRECTORIO, SECUENCIA_P y ORDEN son las llaves que identifican a una persona única
    llaves = ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN']
    df_completo = pd.merge(df_carac, df_ocupados, on=llaves, how='inner')

    print(f"Total de registros tras el cruce: {len(df_completo)}")

    # 3. Filtrado y Limpieza Básica
    # Renombrar columnas clave (Los nombres P... son estándar del DANE, ajustalos según el diccionario del mes)
    columnas_utiles = {
        'P6040': 'edad',
        'P3042': 'nivel_educativo',  # A veces es P3042 o similar dependiendo del año
        'P6430': 'tipo_contrato',
        'RAMA2D_R4': 'sector_economico',
        'INGLABO': 'salario_mensual'  # Variable objetivo
    }

    # Quedarnos solo con las columnas que nos interesan
    columnas_existentes = {k: v for k, v in columnas_utiles.items() if k in df_completo.columns}
    df_modelo = df_completo.rename(columns=columnas_existentes)[list(columnas_existentes.values())]

    # Limpiar valores nulos y ceros (gente que no reportó sueldo)
    df_modelo = df_modelo.dropna(subset=['salario_mensual', 'nivel_educativo'])
    df_modelo = df_modelo[df_modelo['salario_mensual'] > 100000]  # Eliminar sueldos irreales/errores de digitación

    print(f"Total de registros limpios para el modelo: {len(df_modelo)}")

    # Exportar el dataset limpio listo para el XGBoost
    df_modelo.to_csv("dataset_colombia_limpio.csv", index=False)
    print("¡Dataset guardado con éxito!")

    return df_modelo

# Ejemplo de uso (Rutas de los CSVs que descargues)
# df_final = preparar_datos_dane("Características_generales.csv", "Ocupados.csv")