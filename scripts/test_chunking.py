from pathlib import Path
import sys
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ingestion.document_loader import load_directory
from src.ingestion.chunker import chunk_documents


RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    print("=" * 70)
    print("IOCL CHUNKING TEST")
    print("=" * 70)

    records = load_directory(RAW_DATA_DIR)

    print("\nCreating retrieval-ready chunks...")

    chunks = chunk_documents(
        records,
        max_chars=1800,
        overlap_chars=250,
    )

    print("\n" + "=" * 70)
    print("CHUNKING SUMMARY")
    print("=" * 70)

    print(f"Input records: {len(records)}")
    print(f"Generated chunks: {len(chunks)}")

    if not chunks:
        print("No chunks generated.")
        return

    lengths = [len(chunk["text"]) for chunk in chunks]

    print(f"Shortest chunk: {min(lengths)} characters")
    print(f"Longest chunk:  {max(lengths)} characters")
    print(
        f"Average chunk:  "
        f"{sum(lengths) / len(lengths):.1f} characters"
    )

    sources = Counter(
        chunk["metadata"]["source"]
        for chunk in chunks
    )

    print("\nChunks by document:")

    for source, count in sources.most_common():
        print(f"  {count:>3}  {source}")

    print("\n" + "=" * 70)
    print("SAMPLE CHUNKS")
    print("=" * 70)

    for number, chunk in enumerate(chunks[:5], start=1):
        print("\n" + "-" * 70)
        print(f"CHUNK {number}")
        print("-" * 70)

        metadata = chunk["metadata"]

        print(f"Source: {metadata['source']}")

        if "page" in metadata:
            print(f"Page: {metadata['page']}")

        if "slide" in metadata:
            print(f"Slide: {metadata['slide']}")

        if "sheet" in metadata:
            print(f"Sheet: {metadata['sheet']}")

        print(
            f"Chunk: {metadata['chunk_index']}/"
            f"{metadata['chunks_in_record']}"
        )

        print(f"Characters: {metadata['character_count']}")

        print("\nTEXT:")
        print(chunk["text"][:1000])


if __name__ == "__main__":
    main()