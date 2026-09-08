import json
from pathlib import Path


class ExtractionCache:

    def __init__(
        self,
        cache_file: Path,
    ):
        self.cache_file = cache_file

        self.cache_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.processed = self._load()

    def _load(self) -> set[str]:

        if not self.cache_file.exists():
            return set()

        processed = set()

        with self.cache_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                chunk_id = record.get(
                    "chunk_id"
                )

                if chunk_id:
                    processed.add(
                        chunk_id
                    )

        return processed

    def contains(
        self,
        chunk_id: str,
    ) -> bool:
        return chunk_id in self.processed

    def mark_processed(
        self,
        chunk_id: str,
        source: str,
    ):
        if chunk_id in self.processed:
            return

        record = {
            "chunk_id": chunk_id,
            "source": source,
            "status": "success",
        }

        with self.cache_file.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        self.processed.add(
            chunk_id
        )