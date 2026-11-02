"""Document chunking and splitting engine for Enterprise RAG.

Provides recursive character text splitting with header context preservation
and rich metadata tagging for vector and keyword search.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TextChunk:
    chunk_id: str
    content: str
    source: str
    doc_title: str
    chunk_index: int
    header_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class MarkdownTextSplitter:
    """Recursive markdown text splitter preserving section headers and paragraph semantics."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]

    def _generate_chunk_id(self, source: str, index: int, content: str) -> str:
        raw = f"{source}:{index}:{content[:64]}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def split_text(self, text: str, source: str = "document", title: str = "") -> list[TextChunk]:
        """Split text recursively into overlapping semantic chunks."""
        raw_chunks = self._split_recursive(text, self.chunk_size, self.chunk_overlap)
        chunks: list[TextChunk] = []

        current_header = title
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # Detect leading header if present
            header_match = re.search(r"^(#{1,3})\s+(.+)$", chunk_text, re.MULTILINE)
            if header_match:
                current_header = header_match.group(2).strip()

            chunk_id = self._generate_chunk_id(source, idx, chunk_text)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    source=source,
                    doc_title=title or Path(source).stem,
                    chunk_index=idx,
                    header_context=current_header,
                    metadata={
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                    },
                )
            )

        return chunks

    def _split_recursive(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        if len(text) <= chunk_size:
            return [text]

        # Find best separator
        chosen_sep = " "
        for sep in self.separators:
            if sep in text:
                chosen_sep = sep
                break

        splits = text.split(chosen_sep)
        chunks: list[str] = []
        current_chunk = ""

        for part in splits:
            candidate = f"{current_chunk}{chosen_sep}{part}" if current_chunk else part
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Keep overlap from the end of current_chunk
                    overlap_start = max(0, len(current_chunk) - chunk_overlap)
                    overlap_text = current_chunk[overlap_start:]
                    current_chunk = f"{overlap_text}{chosen_sep}{part}" if overlap_text else part
                else:
                    # Part itself exceeds chunk_size, force slice
                    chunks.append(part[:chunk_size])
                    current_chunk = part[chunk_size - chunk_overlap:]

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def split_file(self, file_path: str | Path) -> list[TextChunk]:
        """Read and split a file from filesystem."""
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="ignore")
        title = path.stem.replace("-", " ").replace("_", " ").title()
        return self.split_text(content, source=str(path.name), title=title)
