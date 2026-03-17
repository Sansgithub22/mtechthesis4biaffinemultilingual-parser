# utils/conllu_utils.py
# Lightweight CoNLL-U reader/writer that supports:
#   - Standard 10-column UD format
#   - Multi-word tokens (rows with id like "1-2") — skipped for parsing
#   - Empty nodes (id like "1.1")                 — skipped for parsing
#   - Sentence-level comments (# key = value)

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict


@dataclass
class Token:
    id:     int               # 1-indexed; 0 = ROOT sentinel
    form:   str               # surface form
    lemma:  str    = "_"
    upos:   str    = "_"
    xpos:   str    = "_"
    feats:  str    = "_"
    head:   int    = 0        # 0 = root attachment
    deprel: str    = "root"
    deps:   str    = "_"
    misc:   str    = "_"

    def to_conllu_line(self) -> str:
        return "\t".join([
            str(self.id), self.form, self.lemma, self.upos, self.xpos,
            self.feats, str(self.head), self.deprel, self.deps, self.misc
        ])


@dataclass
class Sentence:
    tokens:   List[Token]            = field(default_factory=list)
    comments: List[str]              = field(default_factory=list)  # raw comment lines

    # ── Convenience helpers ──────────────────────────────────────────────────
    def words(self) -> List[str]:
        return [t.form for t in self.tokens]

    def heads(self) -> List[int]:
        return [t.head for t in self.tokens]

    def deprels(self) -> List[str]:
        return [t.deprel for t in self.tokens]

    def get_comment(self, key: str) -> Optional[str]:
        prefix = f"# {key} = "
        for c in self.comments:
            if c.startswith(prefix):
                return c[len(prefix):].strip()
        return None

    def set_comment(self, key: str, value: str):
        prefix = f"# {key} = "
        for i, c in enumerate(self.comments):
            if c.startswith(prefix):
                self.comments[i] = f"{prefix}{value}"
                return
        self.comments.append(f"{prefix}{value}")

    def to_conllu_block(self) -> str:
        lines = list(self.comments)
        lines += [t.to_conllu_line() for t in self.tokens]
        lines.append("")   # blank separator
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Reader
# ─────────────────────────────────────────────────────────────────────────────
def read_conllu(path: str | Path) -> List[Sentence]:
    """
    Read a CoNLL-U file; return list of Sentence objects.
    Multi-word tokens (id like '1-2') and empty nodes ('1.1') are skipped.
    """
    sentences: List[Sentence] = []
    current   = Sentence()

    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            if not line:                         # blank line → sentence boundary
                if current.tokens:
                    sentences.append(current)
                current = Sentence()
                continue

            if line.startswith("#"):             # comment
                current.comments.append(line)
                continue

            cols = line.split("\t")
            if len(cols) != 10:
                continue

            tok_id = cols[0]
            # Skip multi-word tokens and empty nodes
            if "-" in tok_id or "." in tok_id:
                continue

            current.tokens.append(Token(
                id     = int(tok_id),
                form   = cols[1],
                lemma  = cols[2],
                upos   = cols[3],
                xpos   = cols[4],
                feats  = cols[5],
                head   = int(cols[6]) if cols[6] != "_" else 0,
                deprel = cols[7],
                deps   = cols[8],
                misc   = cols[9],
            ))

    if current.tokens:           # last sentence without trailing blank line
        sentences.append(current)

    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# Writer
# ─────────────────────────────────────────────────────────────────────────────
def write_conllu(sentences: List[Sentence], path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for sent in sentences:
            fh.write(sent.to_conllu_block())
            fh.write("\n")
