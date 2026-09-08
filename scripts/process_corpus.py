from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ingestion.processor import process_corpus


RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)


def main():

    chunks = process_corpus(
        raw_directory=RAW_DIR,
        output_path=OUTPUT_FILE,
    )

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(f"Chunks persisted: {len(chunks)}")

    if chunks:
        example = chunks[0]

        print("\nExample chunk ID:")
        print(example["chunk_id"])

        print("\nSource:")
        print(example["metadata"]["source"])

        print("\nText preview:")
        print(example["text"][:300])


if __name__ == "__main__":
    main()