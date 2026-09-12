"""Angebote aus Dokumenten extrahieren und als AdapterData-JSON ablegen.

Ein Durchlauf: python pipeline.py
Aus dem Notebook: import pipeline; pipeline.run()
"""

import base64
import json
import mimetypes
import os
import re
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# Provider und Modell in einem String. Ein Anbieterwechsel ist genau diese Zeile
# plus das passende Paket, z. B. "anthropic:claude-opus-5" mit langchain-anthropic.
MODEL = "google_genai:gemini-3.6-flash"

# Landet in adapter.name. Konstante der Pipeline, kein Extraktionsergebnis.
ADAPTER_NAME = "parser-notebook"

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")

load_dotenv(dotenv_path=Path.cwd() / ".env")

# Ein Key fuer jeden Anbieter - init_chat_model reicht ihn an die passende
# Provider-Klasse durch, egal ob Google, Anthropic oder OpenAI.
API_KEY = os.environ["LLM_API_KEY"]

CHAT_MODEL = init_chat_model(MODEL, api_key=API_KEY)

# Dauer und Tokenverbrauch des letzten Modellaufrufs, gesetzt von extract_items.
LAST_USAGE: dict = {}


def _angebote(count: int) -> str:
    return f"{count} {'Angebot' if count == 1 else 'Angebote'}"


def _zahl(value: int) -> str:
    return f"{value:,}".replace(",", ".")


# --- Schema ---------------------------------------------------------------

LOCALIZED_TEXT = {
    "type": "object",
    "patternProperties": {"^[a-z]{2}$": {"type": "string"}},
    "additionalProperties": False,
}

ITEM_PROPERTIES = {
    "title": {"type": "string"},
    "titleAddOn": LOCALIZED_TEXT,
    "image": {"type": "string"},
    "state": {"type": "string", "enum": ["public", "draft", "archived", "suggestion"]},
    "tags": {"type": "array", "items": {"type": "string"}},
    "primaryTopic": {"type": "string"},
    "brief": LOCALIZED_TEXT,
    "description": LOCALIZED_TEXT,
    "location_ref": {"type": "string"},
    "location": {"type": "string"},
    "directions": LOCALIZED_TEXT,
    "address": {"type": "string"},
    "zip": {"type": "string"},
    "city": {"type": "string"},
    "latitude": {"type": "number"},
    "longitude": {"type": "number"},
    "recurring_event": {"type": "string"},
    "responsibleInstitution": {"type": "string"},
    "sponsors": {"type": "string"},
    "website": {"type": "string"},
    "email": {"type": "string"},
    "facebook": {"type": "string"},
    "whatsapp": {"type": "string"},
    "contact": {"type": "string"},
    "phone": {"type": "string"},
    "mobile": {"type": "string"},
    "editingNote": {"type": "string"},
    "hours": LOCALIZED_TEXT,
    "accessibility": LOCALIZED_TEXT,
    "charge": LOCALIZED_TEXT,
    "venue": LOCALIZED_TEXT,
    "resubmissionDate": {"type": "string"},
}

# Ein einzelnes Angebot. Nur dieses Schema geht in den Prompt.
ITEM_SCHEMA = {
    "title": "Item",
    "type": "object",
    "properties": ITEM_PROPERTIES,
    "required": ["title", "state", "brief", "description"],
    "additionalProperties": False,
}

# Die Gesamtform, die build_adapter_data erzeugt. Referenz fuer das Zielsystem.
TARGET_SCHEMA = {
    "$id": "/schema",
    "title": "AdapterData",
    "description": "Results of an adapter update including item and meta data.",
    "type": "object",
    "properties": {
        "adapter": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "sourceName": {"type": "string"},
                "sourceUrl": {"type": "string"},
            },
            "required": ["name", "sourceName"],
            "additionalProperties": False,
        },
        "lastUpdate": {"type": "integer"},
        "version": {"type": "string"},
        "itemsRecord": {
            "type": "object",
            "patternProperties": {"^[\\w-]+$": ITEM_SCHEMA},
            "additionalProperties": False,
        },
    },
    "required": ["adapter", "lastUpdate", "itemsRecord"],
    "additionalProperties": False,
}


# --- Prompt ---------------------------------------------------------------

SYSTEM_INSTRUCTION = """Du extrahierst Angebote und Veranstaltungen aus bereitgestellten Dokumenten.

Antworte ausschliesslich mit einem gueltigen JSON-Objekt, ohne Markdown und ohne Erklaerung.
Die Keys sind sprechende Slugs (Kleinbuchstaben, Ziffern, Bindestriche), die Werte folgen
exakt dem unten stehenden Item-Schema.

Regeln zur Abgrenzung:
- Lege fuer jedes eigenstaendige Angebot und jede eigenstaendige Veranstaltung einen
  separaten Eintrag an. Fasse mehrere Angebote nicht zu einem Eintrag zusammen.
- Ein Angebot mit mehreren Terminen bleibt EIN Eintrag. Nenne alle Termine in "hours",
  regelmaessige Wiederholungen zusaetzlich in "recurring_event".

Regeln zum Inhalt:
- Verwende null, wenn ein Wert nicht erkennbar ist.
- Gib keine Informationen hinzu, die nicht im Dokument belegt sind.

Item-Schema:
{schema}
"""


# --- Extraktion -----------------------------------------------------------


def _content_block(path: Path) -> dict:
    """Eine Datei als LangChain-Content-Block."""
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {
        "type": "image" if media_type.startswith("image/") else "file",
        "base64": base64.b64encode(path.read_bytes()).decode(),
        "mime_type": media_type,
    }


def _slugify(key: str) -> str:
    """Modell-Key saeubern - er wird itemsRecord-Key und Dateiname."""
    return re.sub(r"[^\w-]+", "-", key.strip().lower()).strip("-") or "angebot"


def extract_items(paths: list[Path]) -> dict:
    """Alle Angebote aus den Dateien holen, in einem Modellaufruf.

    Schreibt Dauer und Tokenverbrauch des Aufrufs nach LAST_USAGE.
    """
    blocks = []
    for path in paths:
        blocks.append(_content_block(path))
        blocks.append({"type": "text", "text": f"Dateiname: {path.name}"})
    blocks.append(
        {
            "type": "text",
            "text": SYSTEM_INSTRUCTION.format(
                schema=json.dumps(ITEM_SCHEMA, ensure_ascii=False, indent=2)
            ),
        }
    )

    started = time.monotonic()
    response = CHAT_MODEL.invoke([HumanMessage(content=blocks)])
    LAST_USAGE.update(response.usage_metadata or {})
    LAST_USAGE["seconds"] = time.monotonic() - started

    text = response.text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```").removeprefix("json").removesuffix("```").strip()
    return json.loads(text)


def build_adapter_data(slug: str, item: dict, source_name: str) -> dict:
    """Ein Angebot als eigenstaendige AdapterData-Instanz verpacken."""
    return {
        "adapter": {"name": ADAPTER_NAME, "sourceName": source_name},
        "lastUpdate": int(time.time()),
        "itemsRecord": {slug: item},
    }


def extract_file(path: Path) -> list[dict]:
    """Eine Datei, ein Modellaufruf, je gefundenem Angebot eine AdapterData-Instanz.

    Zusammengehoerende Dateien (Vorder- und Rueckseite eines Plakats) stattdessen
    gemeinsam an extract_items geben.
    """
    items = extract_items([path])
    return [
        build_adapter_data(_slugify(key), item, path.name) for key, item in items.items()
    ]


def extract_from_paths(file_paths: list[Path]) -> list[dict]:
    """Alle Dateien extrahieren, ohne zu schreiben. Ein Modellaufruf je Datei."""
    return [result for path in file_paths for result in extract_file(path)]


def write_result(result: dict, output_dir: Path = OUTPUT_DIR) -> Path:
    """Ein Angebot als JSON-Datei, in einem Unterordner je Quelldatei."""
    target_dir = output_dir / _slugify(Path(result["adapter"]["sourceName"]).stem)
    target_dir.mkdir(parents=True, exist_ok=True)

    slug = next(iter(result["itemsRecord"]))
    target = target_dir / f"{slug}.json"
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def run(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR) -> list[dict]:
    """Ein kompletter Durchlauf: je Datei extrahieren und sofort schreiben.

    Geschrieben wird nach jedem Modellaufruf, nicht erst am Ende - bricht ein
    spaeterer Aufruf ab, bleiben die bereits erzeugten Angebote erhalten.
    Der Ausgabeordner wird dafuer erst geleert, wenn der erste Aufruf steht.
    """
    file_paths = [path for path in sorted(input_dir.glob("*")) if path.is_file()]
    print(f"Modell:  {MODEL}")
    print(f"Eingabe: {input_dir}/ ({len(file_paths)} Dateien)")
    print(f"Ausgabe: {output_dir}/\n")

    started = time.monotonic()
    results = []
    tokens_in = tokens_out = 0
    cleared = False

    for number, path in enumerate(file_paths, start=1):
        print(f"[{number}/{len(file_paths)}] {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        extracted = extract_file(path)

        tokens_in += LAST_USAGE.get("input_tokens", 0)
        tokens_out += LAST_USAGE.get("output_tokens", 0)

        # Erst aufraeumen, wenn der erste Aufruf durch ist - ein Fehlschlag
        # laesst den vorherigen Lauf damit unangetastet.
        if not cleared:
            shutil.rmtree(output_dir, ignore_errors=True)
            output_dir.mkdir(parents=True)
            cleared = True

        target_dir = None
        for result in extracted:
            target_dir = write_result(result, output_dir).parent
        results.extend(extracted)

        print(
            f"        {_angebote(len(extracted))}"
            f" | {LAST_USAGE.get('seconds', 0):.0f}s"
            f" | Tokens {_zahl(LAST_USAGE.get('input_tokens', 0))} ein"
            f" / {_zahl(LAST_USAGE.get('output_tokens', 0))} aus"
        )
        if target_dir:
            print(f"        -> {target_dir}")

    print(
        f"\nFertig: {_angebote(len(results))} aus {len(file_paths)} Dateien"
        f" | {time.monotonic() - started:.0f}s"
        f" | Tokens {_zahl(tokens_in)} ein / {_zahl(tokens_out)} aus"
    )
    return results


if __name__ == "__main__":
    run()
