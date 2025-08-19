
class DocxLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def load(self) -> Tuple[List[Chunk], List[str]]:
        chunks: List[Chunk] = []
        doc_names: List[str] = []
        cid = 0
        for di, path in enumerate(sorted(self.data_dir.glob("*.docx"))):
            doc = Document(str(path))
            shortname = self._shortname_from_filename(path.name)
            doc_names.append(shortname)

            # paragraphs → sentence-aware chunking
            para_texts = [normalize_text(p.text) for p in doc.paragraphs if p.text and p.text.strip()]
            for ch_text in self._word_aware_chunk("\n".join(para_texts)):
                loc = extract_locator(ch_text)
                chunks.append(Chunk(id=cid, doc_id=di, doc_name=shortname, text=ch_text, kind="para", locator=loc))
                cid += 1

            # tables → row-level chunks (attach header as context)
            for tbl in doc.tables:
                header = None
                if len(tbl.rows) > 0:
                    header = " | ".join(cell.text.strip() for cell in tbl.rows[0].cells)
                    header = normalize_text(header)
                for ri, row in enumerate(tbl.rows):
                    if ri == 0:
                        # also create a header chunk (helps retrieval)
                        if header and header.strip():
                            loc = extract_locator(header)
                            chunks.append(Chunk(id=cid, doc_id=di, doc_name=shortname, text=f"[Таблица]\n{header}", kind="table", locator=loc))
                            cid += 1
                        continue
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    row_text = normalize_text(row_text)
                    combined = (f"[Таблица]\n{header}\n{row_text}" if header else f"[Таблица]\n{row_text}")
                    loc = extract_locator(combined)
                    chunks.append(Chunk(id=cid, doc_id=di, doc_name=shortname, text=combined, kind="table", locator=loc))
                    cid += 1
        return chunks, doc_names

    @staticmethod
    def _shortname_from_filename(fname: str) -> str:
        # e.g. "СП 20.13330.2016 Нагрузки...docx" → "СП 20.13330.2016"
        base = Path(fname).stem
        m = re.match(r"^([^\s]+\s*[^\s]+)\b", base)
        return m.group(1) if m else base

    @staticmethod
    def _word_aware_chunk(text: str, max_chars: int = 1600, overlap_sents: int = 1) -> List[str]:
        """Sentence-based sliding window, keeping word boundaries.
        - Split by sentences (razdel)
        - Accumulate into windows up to max_chars
        - Overlap by N sentences to preserve context
        """
        sents = [s.text.strip() for s in sentenize(text) if s.text and s.text.strip()]
        res = []
        i = 0
        while i < len(sents):
            window = []
            total = 0
            j = i
            while j < len(sents) and total + len(sents[j]) + 1 <= max_chars:
                window.append(sents[j])
                total += len(sents[j]) + 1
                j += 1
            if window:
                res.append(" ".join(window))
            if j == i:  # single very long sentence, hard-cut at word boundary
                toks = [t.text for t in tokenize(sents[i])]
                buf = []
                tlen = 0
                for tok in toks:
                    if tlen + len(tok) + 1 > max_chars:
                        break
                    buf.append(tok)
                    tlen += len(tok) + 1
                if buf:
                    res.append(" ".join(buf))
                j = i + 1
            i = max(j - overlap_sents, j)
        return res