"""Prompts. Kept deliberately plain for the Phase 0 kill test."""

from .data import LANGUAGE_NAMES

SYSTEM = "You are an expert in correcting OCR errors in historical documents."


def _describe(meta: dict) -> str:
    lang = LANGUAGE_NAMES.get(meta.get("language", ""), meta.get("language", "unknown language"))
    doc_type = meta.get("document_type") or "document"
    date = meta.get("date")
    when = f", {date}" if date and date != "n/a" else ""
    return f"a historical {lang} {doc_type}{when}"


def zeroshot_messages(meta: dict, ocr: str) -> list[dict]:
    user = (
        f"The following text from {_describe(meta)} was produced by OCR and contains OCR errors.\n"
        "Correct the OCR errors. Keep the original historical spelling, wording, punctuation and line "
        "breaks wherever they are not OCR errors. Do not modernise, paraphrase, translate, or add content; "
        "if the text is cut off, do not complete it.\n"
        "Output only the corrected text, with no explanations.\n\n"
        f"Text:\n{ocr}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


OPEN, CLOSE = "⟦", "⟧"


def span_messages(meta: dict, left: str, span: str, right: str) -> list[dict]:
    user = (
        f"The following excerpt from {_describe(meta)} was produced by OCR. The passage between "
        f"{OPEN} and {CLOSE} contains at least one OCR error.\n"
        "Rewrite ONLY the marked passage with its OCR errors corrected. Keep the original historical "
        "spelling, capitalisation and punctuation where they are not OCR errors; do not modernise. "
        "The surrounding text is context only and must not be repeated.\n"
        "Answer with the corrected passage enclosed in <fix></fix> tags and nothing else.\n\n"
        f"Excerpt:\n{left}{OPEN}{span}{CLOSE}{right}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def window_messages(meta: dict, left: str, span: str, right: str) -> list[dict]:
    """Phase 0b window interface: the model rewrites a short window (about +-40 chars) around the
    marked span; the caller aligns the rewrite to the window and keeps only edits inside the span."""
    user = (
        f"The following short excerpt from {_describe(meta)} was produced by OCR. The passage between "
        f"{OPEN} and {CLOSE} contains at least one OCR error. The excerpt may start or end in the middle "
        "of a sentence or word; do not complete it.\n"
        "Rewrite the whole excerpt with its OCR errors corrected. Keep the original historical spelling, "
        "capitalisation and punctuation where they are not OCR errors; do not modernise, paraphrase, "
        "translate, add or remove content.\n"
        f"Answer with the corrected excerpt (without the {OPEN} {CLOSE} markers) enclosed in <fix></fix> "
        "tags and nothing else.\n\n"
        f"Excerpt:\n{left}{OPEN}{span}{CLOSE}{right}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
