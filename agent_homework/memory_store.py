import json
from pathlib import Path

from datetime import datetime, timezone
MEMORY_FILE = Path(__file__).with_name(
    "long_term_memory.json"
)


def load_memories() -> dict:
    if not MEMORY_FILE.exists():
        return {}

    with MEMORY_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_memories(memories: dict) -> None:
    with MEMORY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            memories,
            file,
            ensure_ascii=False,
            indent=2,
        )


def remember(key: str, value: str) -> None:
    memories = load_memories()

    memories[key] = {
        "value": value,
        "source": "user_explicit",
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    save_memories(memories)


def recall(key: str) -> dict | None:
    memories = load_memories()
    record = memories.get(key)
    if record is None:
        return None
    # 兼容前面实验留下的旧字符串格式
    if isinstance(record, str):
        return {
            "value": record,
            "source": "legacy",
            "updated_at": None,
        }

    return record

def search_memories(keyword: str) -> list[dict]:
    keyword = keyword.strip().lower()

    if not keyword:
        return []

    memories = load_memories()
    results = []

    for key, raw_record in memories.items():
        if isinstance(raw_record, str):
            record = {
                "value": raw_record,
                "source": "legacy",
                "updated_at": None,
            }
        else:
            record = raw_record

        searchable_text = (
            f"{key} {record['value']}"
        ).lower()

        if keyword in searchable_text:
            results.append(
                {
                    "key": key,
                    "value": record["value"],
                    "source": record["source"],
                    "updated_at": record["updated_at"],
                }
            )

    return results

def forget(key: str) -> bool:
    memories = load_memories()

    if key not in memories:
        return False

    del memories[key]
    save_memories(memories)

    return True

def main():
    print(
        "Before:",
        recall("research_field"),
    )

    remember(
        "research_field",
        "stereo matching",
    )

    print(
        "After:",
        recall("research_field"),
    )


if __name__ == "__main__":
    main()