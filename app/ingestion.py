"""Document parsing and section-aware, bounded character chunking."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from app.errors import InvalidDocument

ALLOWED_EXTENSIONS = {".md", ".txt", ".html", ".htm"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    title: str
    source: str
    section: str
    position: int
    distance: float = 0.0


def parse_document(filename: str, data: bytes) -> tuple[str, str]:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise InvalidDocument("Supported formats: .md, .txt, .html, .htm")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InvalidDocument("Documents must use UTF-8 encoding") from exc
    if "\x00" in text:
        raise InvalidDocument("Binary documents are not supported")
    title = Path(filename).stem
    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(text, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        root = soup.find("article") or soup.find("main") or soup
        for element in root.select("script, style, nav, footer, header"):
            element.decompose()
        # Preserve code blocks and heading boundaries for the same Markdown splitter.
        for pre in root.find_all("pre"):
            pre.replace_with("\n\n```\n" + pre.get_text() + "\n```\n\n")
        for heading in root.find_all(re.compile(r"^h[1-6]$")):
            heading.replace_with(
                "\n\n"
                + "#" * int(heading.name[1])
                + " "
                + heading.get_text(" ", strip=True)
                + "\n\n"
            )
        text = root.get_text("\n")
    else:
        match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if match:
            title = match.group(1).strip()
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()
    if not text:
        raise InvalidDocument("Document has no indexable text")
    return title[:300], text


def split_document(
    text: str, title: str, source: str, chunk_size: int, overlap: int
) -> tuple[str, list[Chunk]]:
    """Prefer headings and paragraph ends; hard-split oversized blocks as a last resort.

    Heading detection ignores fenced code. Overlap stays within each section.
    Character limits are intentional; these are not token counts.
    """
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size")
    document_id = hashlib.sha256((source + "\0" + text).encode()).hexdigest()[:24]
    sections: list[tuple[str, str]] = []
    heading = title
    lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        was_fenced = fence is not None
        if marker:
            current = marker.group(1)[0]
            if fence is None:
                fence = current
            elif current == fence:
                fence = None
        match = re.match(r"^#{1,6}\s+(.+)$", line) if not was_fenced else None
        if match:
            if "\n".join(lines).strip():
                sections.append((heading, "\n".join(lines).strip()))
            heading = match.group(1).strip()
            lines = [line]
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        sections.append((heading, "\n".join(lines).strip()))

    chunks: list[Chunk] = []
    for section, content in sections:
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            if end < len(content):
                # Never choose a boundary before the overlap, which would prevent progress.
                floor = start + max(chunk_size // 2, overlap + 1)
                boundary = content.rfind("\n\n", floor, end)
                if boundary < floor:
                    boundary = content.rfind("\n", floor, end)
                if boundary >= floor:
                    end = boundary
            piece = content[start:end].strip()
            if piece:
                position = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}:{position}",
                        document_id=document_id,
                        text=piece,
                        title=title,
                        source=source,
                        section=section[:300],
                        position=position,
                    )
                )
            if end == len(content):
                break
            start = end - overlap
    return document_id, chunks
