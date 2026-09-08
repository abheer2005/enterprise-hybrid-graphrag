from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ingestion.document_loader import load_directory


RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    print("=" * 70)
    print("IOCL DOCUMENT INGESTION TEST")
    print("=" * 70)

    records = load_directory(RAW_DATA_DIR)

    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)

    print(f"Total extracted records: {len(records)}")

    total_characters = sum(
        len(record["text"])
        for record in records
    )

    print(f"Total extracted characters: {total_characters:,}")

    print("\nFIRST 3 RECORDS:\n")

    for number, record in enumerate(records[:3], start=1):
        print("-" * 70)
        print(f"RECORD {number}")
        print(f"Metadata: {record['metadata']}")
        print("\nText preview:")

        preview = record["text"][:500]
        print(preview)

        print()


if __name__ == "__main__":
    main()