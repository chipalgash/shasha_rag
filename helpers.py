def extract_json(s: str) -> str:
    """Extract the first JSON object from a string"""
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return s


def unique_citations(chunks: List[Chunk], ids: List[int]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    id2chunk = {c.id: c for c in chunks}
    for cid in ids:
        c = id2chunk.get(cid)
        if not c:
            continue
        key = (c.doc_name, c.locator.get("clause"), c.locator.get("table"))
        if key in seen:
            continue
        seen.add(key)
        item = {"document": c.doc_name}
        if c.locator.get("clause"):
            item["clause"] = c.locator["clause"]
        if c.locator.get("table"):
            item["table"] = c.locator["table"]
        out.append(item)
    return out
