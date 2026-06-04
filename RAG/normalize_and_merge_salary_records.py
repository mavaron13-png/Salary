
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_LHH_PATH = Path(r"C:\Users\ma_va\Documents\Salary\RAG\estudio_laboral_lhh_records.jsonl")
DEFAULT_MYDNA_PATH = Path(r"C:\Users\ma_va\Documents\Salary\RAG\mydna_colombia_records.jsonl")
DEFAULT_OUTPUT_PATH = Path(r"C:\Users\ma_va\Documents\Salary\RAG\salary_records_normalized_merged.jsonl")
DEFAULT_REPORT_PATH = Path(r"C:\Users\ma_va\Documents\Salary\RAG\salary_records_normalized_report.json")


CANONICAL_FIELDS = [
    "id",
    "original_id",
    "schema_version",
    "pais",
    "fuente",
    "archivo_fuente",
    "sector_area",
    "tabla_id",
    "tabla_titulo",
    "cargo",
    "tamano_empresa",
    "facturacion_anual",
    "salario_minimo_cop",
    "salario_medio_cop",
    "salario_maximo_cop",
    "salario_anual_cop",
    "variable",
    "salario_raw",
    "salary_parse_kind",
    "moneda",
    "unidad",
    "texto_rag",
    "metadata",
]


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    return text if text else None


def normalize_int(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    text = str(value).strip()

    if not text or text.upper() in {"N/A", "NA", "NULL", "NONE", "-"}:
        return None

    digits = re.sub(r"[^\d-]", "", text)

    if digits in {"", "-"}:
        return None

    return int(digits)


def slugify(value: str | None) -> str:
    if not value:
        return "unknown"

    replacements = str.maketrans(
        "áéíóúÁÉÍÓÚñÑüÜ",
        "aeiouAEIOUnNuU",
    )

    value = value.translate(replacements).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")

    return value or "unknown"


def detect_source_type(record: dict[str, Any]) -> str:
    fuente = (record.get("fuente") or "").lower()
    record_id = (record.get("id") or "").lower()

    if "mydna" in fuente or record_id.startswith("colombia-"):
        return "mydna"

    if "lhh" in fuente or record_id.startswith("lhh-"):
        return "lhh"

    if "cargo" in record and "area" in record:
        return "mydna"

    if "puesto" in record and "tabla_titulo" in record:
        return "lhh"

    return "unknown"


def build_canonical_id(record: dict[str, Any], source_type: str) -> str:
    original_id = normalize_text(record.get("id"))

    if original_id:
        return original_id

    pais = normalize_text(record.get("pais")) or "Colombia"
    fuente = normalize_text(record.get("fuente")) or source_type
    cargo = normalize_text(record.get("cargo") or record.get("puesto")) or "cargo"
    sector_area = normalize_text(record.get("area") or record.get("tabla_titulo")) or "sector"
    tamano_empresa = normalize_text(record.get("tamano_empresa")) or "tamano"

    return "-".join(
        [
            slugify(pais),
            slugify(fuente),
            slugify(sector_area),
            slugify(tamano_empresa),
            slugify(cargo),
        ]
    )


def normalize_salary_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Schema canónico para unir MyDNA y LHH.

    MyDNA trae:
        cargo, area, facturacion_anual, salario_medio_cop, salario_anual_cop, variable

    LHH trae:
        puesto, tabla_titulo, tabla_id, salario_raw, salary_parse_kind

    Esta función unifica:
        puesto -> cargo
        area / tabla_titulo -> sector_area
        campos faltantes -> None
    """

    source_type = detect_source_type(record)

    cargo = normalize_text(record.get("cargo") or record.get("puesto"))
    sector_area = normalize_text(record.get("area") or record.get("tabla_titulo"))

    original_id = normalize_text(record.get("id"))
    canonical_id = build_canonical_id(record, source_type)

    normalized = {
        "id": canonical_id,
        "original_id": original_id,
        "schema_version": "salary_record_normalized_v1",

        "pais": normalize_text(record.get("pais")) or "Colombia",
        "fuente": normalize_text(record.get("fuente")),
        "archivo_fuente": normalize_text(record.get("archivo_fuente")),

        "sector_area": sector_area,
        "tabla_id": normalize_text(record.get("tabla_id")),
        "tabla_titulo": normalize_text(record.get("tabla_titulo") or record.get("area")),
        "cargo": cargo,

        "tamano_empresa": normalize_text(record.get("tamano_empresa")),
        "facturacion_anual": normalize_text(record.get("facturacion_anual")),

        "salario_minimo_cop": normalize_int(record.get("salario_minimo_cop")),
        "salario_medio_cop": normalize_int(record.get("salario_medio_cop")),
        "salario_maximo_cop": normalize_int(record.get("salario_maximo_cop")),
        "salario_anual_cop": normalize_int(record.get("salario_anual_cop")),

        "variable": normalize_int(record.get("variable")),
        "salario_raw": normalize_text(record.get("salario_raw")),
        "salary_parse_kind": normalize_text(record.get("salary_parse_kind")),

        "moneda": normalize_text(record.get("moneda")) or "COP",
        "unidad": normalize_text(record.get("unidad")),
        "texto_rag": normalize_text(record.get("texto_rag")),

        "metadata": {
            "source_type": source_type,
            "has_salario_medio": record.get("salario_medio_cop") is not None,
            "has_salario_anual": record.get("salario_anual_cop") is not None,
            "has_variable": record.get("variable") is not None,
            "has_salario_raw": record.get("salario_raw") is not None,
        },
    }

    # Garantiza orden estable de llaves.
    return {field: normalized.get(field) for field in CANONICAL_FIELDS}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    records = []
    bad_lines = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                bad_lines.append(
                    {
                        "file": str(path),
                        "line_number": line_number,
                        "error": str(exc),
                        "line_preview": line[:300],
                    }
                )

    if bad_lines:
        raise ValueError(
            "Hay líneas corruptas en el JSONL. Primer error: "
            + json.dumps(bad_lines[0], ensure_ascii=False)
        )

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(
    path: Path,
    *,
    input_counts: dict[str, int],
    normalized_records: list[dict[str, Any]],
    duplicate_ids: list[str],
) -> None:
    source_counter = Counter(record["fuente"] for record in normalized_records)
    source_type_counter = Counter(record["metadata"]["source_type"] for record in normalized_records)
    sector_counter = Counter(record["sector_area"] for record in normalized_records)

    null_counts = {
        field: sum(1 for record in normalized_records if record.get(field) is None)
        for field in CANONICAL_FIELDS
        if field != "metadata"
    }

    report = {
        "total_input_records": sum(input_counts.values()),
        "input_counts": input_counts,
        "total_normalized_records": len(normalized_records),
        "source_counts": dict(source_counter),
        "source_type_counts": dict(source_type_counter),
        "top_20_sector_area_counts": dict(sector_counter.most_common(20)),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids[:100],
        "null_counts_by_field": null_counts,
        "schema_fields": CANONICAL_FIELDS,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_and_normalize(lhh_path: Path, mydna_path: Path, output_path: Path, report_path: Path) -> None:
    lhh_records = read_jsonl(lhh_path)
    mydna_records = read_jsonl(mydna_path)

    normalized_records = []

    for record in lhh_records:
        normalized_records.append(normalize_salary_record(record))

    for record in mydna_records:
        normalized_records.append(normalize_salary_record(record))

    ids = [record["id"] for record in normalized_records]
    id_counter = Counter(ids)
    duplicate_ids = sorted([record_id for record_id, count in id_counter.items() if count > 1])

    # Si hay IDs duplicados, no los piso. Les agrego sufijo técnico.
    # Mongo y Pinecone odian los duplicados. Bueno, también los humanos, pero los humanos cobran consultoría.
    if duplicate_ids:
        seen = Counter()

        for record in normalized_records:
            record_id = record["id"]
            seen[record_id] += 1

            if id_counter[record_id] > 1:
                record["id"] = f"{record_id}__dup_{seen[record_id]}"

    write_jsonl(output_path, normalized_records)

    write_report(
        report_path,
        input_counts={
            str(lhh_path): len(lhh_records),
            str(mydna_path): len(mydna_records),
        },
        normalized_records=normalized_records,
        duplicate_ids=duplicate_ids,
    )

    print("✅ Normalización completada.")
    print(f"📥 LHH records: {len(lhh_records)}")
    print(f"📥 MyDNA records: {len(mydna_records)}")
    print(f"📦 Total normalizados: {len(normalized_records)}")
    print(f"🧬 Duplicados detectados antes de sufijo: {len(duplicate_ids)}")
    print(f"📄 Output JSONL: {output_path}")
    print(f"📊 Reporte: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normaliza y une registros salariales LHH + MyDNA en un JSONL canónico."
    )

    parser.add_argument("--lhh", default=str(DEFAULT_LHH_PATH), help="Ruta al JSONL de records LHH.")
    parser.add_argument("--mydna", default=str(DEFAULT_MYDNA_PATH), help="Ruta al JSONL de records MyDNA.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH), help="Ruta de salida JSONL normalizado.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Ruta del reporte JSON.")

    args = parser.parse_args()

    merge_and_normalize(
        lhh_path=Path(args.lhh),
        mydna_path=Path(args.mydna),
        output_path=Path(args.out),
        report_path=Path(args.report),
    )


if __name__ == "__main__":
    main()
