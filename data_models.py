@dataclass
class Chunk:
    id: int
    doc_id: int
    doc_name: str
    text: str
    kind: str  # "para" | "table"
    locator: Dict[str, Optional[str]]  # {"clause": str|None, "table": str|None}

    def short_citation(self) -> str:
        parts = [self.doc_name]
        if self.locator.get("clause"):
            parts.append(self.locator["clause"])
        if self.locator.get("table"):
            parts.append(self.locator["table"])
        return ", ".join(parts)