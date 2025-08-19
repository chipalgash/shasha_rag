CLAUSE_RE = re.compile(r"(?:п\.|пункт)\s*\d+(?:\.\d+)*", re.IGNORECASE)
TABLE_RE = re.compile(r"(?:таблица|табл\.)\s*\d+(?:\.\d+)*", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"[ \t]+")


def normalize_text(raw: str) -> str:
    # Keep digits & punctuation. Only collapse whitespace and trim.
    t = MULTISPACE_RE.sub(" ", raw)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def extract_locator(text: str) -> Dict[str, Optional[str]]:
    clause = None
    table = None
    m1 = CLAUSE_RE.search(text)
    if m1:
        clause = m1.group(0)
    m2 = TABLE_RE.search(text)
    if m2:
        table = m2.group(0)
    return {"clause": clause, "table": table}