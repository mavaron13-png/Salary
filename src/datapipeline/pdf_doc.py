import argparse
import sys
from pdf2docx import Converter


def convertir_pdf(ruta_pdf, ruta_docx):
    try:
        print(f"🔄 Iniciando conversión...\n📄 Origen: {ruta_pdf}\n📝 Destino: {ruta_docx}")
        cv = Converter(ruta_pdf)
        cv.convert(ruta_docx)
        cv.close()
        print("✅ ¡Conversión exitosa!")
    except Exception as e:
        print(f"❌ Ocurrió un error durante la conversión: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Configurar los argumentos de consola
    parser = argparse.ArgumentParser(description="Convierte un documento PDF a formato editable DOCX.")
    parser.add_argument("pdf", help="Ruta completa o relativa del archivo PDF de entrada")
    parser.add_argument("docx", help="Ruta completa o relativa del archivo DOCX de salida")

    args = parser.parse_args()

    # Ejecutar la función con los argumentos capturados
    convertir_pdf(args.pdf, args.docx)