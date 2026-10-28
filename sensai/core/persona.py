import json
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = ("name", "role", "tone", "scope")


class PersonaError(Exception):
    pass


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    role: str
    tone: str
    scope: str
    instructions: str = ""

    def system_prompt(self) -> str:
        lines = [f"You are {self.name}, {self.role}", f"Tone: {self.tone}", f"Scope: {self.scope}"]
        if self.instructions:
            lines.append(self.instructions)
        return "\n".join(lines)


def load_persona(path: Path) -> Persona:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PersonaError(f"Cannot read persona file {path}: {e}") from e
    if not isinstance(data, dict):
        raise PersonaError(f"Persona file {path} must contain a JSON object")
    missing = [name for name in REQUIRED_FIELDS if not isinstance(data.get(name), str) or not data[name].strip()]
    if missing:
        raise PersonaError(f"Persona file {path} is missing: {', '.join(missing)}")
    instructions = data.get("instructions", "")
    if not isinstance(instructions, str):
        raise PersonaError(f"Persona file {path}: 'instructions' must be a string")
    return Persona(path.stem, *(data[name].strip() for name in REQUIRED_FIELDS), instructions.strip())


def load_personas(directory: Path) -> dict[str, Persona]:
    if not directory.is_dir():
        return {}
    return {path.stem: load_persona(path) for path in sorted(directory.glob("*.json"))}
