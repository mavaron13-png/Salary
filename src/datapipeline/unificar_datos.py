import pandas as pd
import os

# Definir la ruta exacta de tus datos
data_dir = r"C:\Users\ma_va\Documents\Salary\data\raw\DANE"


def unificar_archivos(prefijo_entrada, nombre_salida):
    print(f"🚀 Iniciando unificación para: {prefijo_entrada}...")
    lista_dfs = []

    # Iterar del mes 1 al 12
    for i in range(1, 13):
        file_name = f"{prefijo_entrada}_{i}.csv"
        file_path = os.path.join(data_dir, file_name)

        if os.path.exists(file_path):
            print(f"  -> Cargando {file_name}...")
            try:
                # El DANE usa ';' como separador y codificación latin-1 para caracteres especiales
                df = pd.read_csv(file_path, sep=';', encoding='latin-1', low_memory=False)
                lista_dfs.append(df)
            except Exception as e:
                print(f"  ❌ Error leyendo {file_name}: {e}")
        else:
            print(f"  ⚠️ Archivo no encontrado: {file_name}")

    if lista_dfs:
        print(f"⏳ Concatenando {len(lista_dfs)} meses en memoria...")
        # Unir todos los DataFrames verticalmente
        df_final = pd.concat(lista_dfs, ignore_index=True)

        # Ruta de guardado
        output_path = os.path.join(data_dir, nombre_salida)

        # Exportar a CSV estandarizado (ahora en utf-8 para no sufrir más adelante)
        print("💾 Guardando el archivo unificado en disco...")
        df_final.to_csv(output_path, sep=';', index=False, encoding='utf-8')

        print(f"✅ ¡Éxito! Archivo guardado: {nombre_salida}")
        print(f"📊 Total de registros consolidados: {len(df_final):,}\n")
    else:
        print(f"❌ No se encontraron archivos con el prefijo '{prefijo_entrada}'.\n")


if __name__ == "__main__":
    # 1. Unificar Características Generales
    unificar_archivos("Características generales", "caracteristicas_generales_2025.csv")

    # 2. Unificar Ocupados
    unificar_archivos("Ocupados", "ocupados_2025.csv")