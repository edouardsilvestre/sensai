import json

ENTITY_TYPES = ("person", "company", "place", "other")

# Sent as Ollama's `format` by /json: the answer is data a program can read, not prose.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    # The numbers of the excerpts that support the claim, as in [2] in a prose answer.
                    "sources": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "sources"],
            },
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                },
                "required": ["name", "type"],
            },
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["date", "description"],
            },
        },
    },
    "required": ["answer", "claims", "entities", "events"],
}

TYPES = {"object": dict, "array": list, "string": str, "integer": int}


class StructuredError(ValueError):
    pass


def schema_errors(value, schema: dict, path: str = "$") -> list[str]:
    """What does not follow the schema, for the subset of JSON schema the schemas of Sensai use."""
    kind = schema["type"]
    if not isinstance(value, TYPES[kind]) or isinstance(value, bool):
        return [f"{path} is not of type {kind}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path} is not one of {', '.join(schema['enum'])}"]
    if kind == "object":
        errors = [f"{path}.{name} is missing" for name in schema.get("required", []) if name not in value]
        for name, field in schema.get("properties", {}).items():
            if name in value:
                errors += schema_errors(value[name], field, f"{path}.{name}")
        return errors
    if kind == "array":
        return [error for i, item in enumerate(value) for error in schema_errors(item, schema["items"], f"{path}[{i}]")]
    return []


def parse_answer(reply: str) -> dict:
    """
    The answer of /json. Ollama already constrains it to ANSWER_SCHEMA, but it is checked again: a guardrail can cut
    it, and another server could ignore the schema. Raises StructuredError when it does not follow it.
    """
    try:
        data = json.loads(reply)
    except json.JSONDecodeError as e:
        raise StructuredError(f"the answer is not valid JSON ({e.msg})") from None
    errors = schema_errors(data, ANSWER_SCHEMA)
    if errors:
        raise StructuredError("the answer does not follow the schema: " + "; ".join(errors[:3]))
    return data


def cited_sources(data: dict) -> list[int]:
    """The excerpt numbers the claims cite, in the order they are first cited."""
    return list(dict.fromkeys(number for claim in data["claims"] for number in claim["sources"]))
