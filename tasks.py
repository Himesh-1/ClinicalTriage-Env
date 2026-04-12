# tasks.py
import json
from pathlib import Path

_TASKS_DIR = Path(__file__).parent / "tasks"


def get_task(task_name: str) -> dict:
    path = _TASKS_DIR / f"{task_name}.json"
    if not path.exists():
        raise ValueError(
            f"Unknown task: {task_name!r}. Available: {list_tasks()}"
        )
    return json.loads(path.read_text())


def list_tasks() -> list[str]:
    return [p.stem for p in sorted(_TASKS_DIR.glob("*.json"))]
