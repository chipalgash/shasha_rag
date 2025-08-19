ANSWER_SYSTEM_PROMPT = (
    "Ты — строгий помощник по нормативной документации. Отвечай только по приведённому контексту. "
    "Формат ответа — JSON с полями: {\"answer\": str, \"used_chunks\": [int, ...]}. "
    "Требования: 1) ответ кратко (1–2 предложения), 2) по-русски, 3) при наличии укажи номера пунктов/таблиц в самом ответе, "
    "4) если данных недостаточно — верни 'Недостаточно информации'."
)

ANSWER_USER_PROMPT_TEMPLATE = (
    "Вопрос: {question}\n\nКонтекст (список чанков):\n{context}\n\n"
    "Верни строго JSON без лишнего текста."
)

def build_context(chunks: List[Chunk]) -> str:
    lines = []
    for c in chunks:
        head = f"[ID={c.id}] {c.short_citation()}"
        body = c.text
        lines.append(f"{head}\n{body}\n")
    return "\n".join(lines)

class RAGEngine:
    def __init__(self, data_dir: Path, backend: str = "ollama", model: str = "Qwen2.5:7b", device: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.backend_kind = backend
        self.model_name = model
        self.device = device

        loader = DocxLoader(self.data_dir)
        self.chunks, self.doc_names = loader.load()
        self.index = HybridIndex(self.chunks)
        self.reranker = Reranker(device=device)
        if backend == "ollama":
            self.llm: LLMBackend = OllamaBackend(model=model)
        elif backend == "hf":
            self.llm = HFLocalBackend(model=model, load_4bit=(device == "cuda"))
        else:
            raise ValueError("backend must be 'ollama' or 'hf'")

    def answer_one(self, question: str, top_k_ctx: int = 8) -> Dict[str, Any]:
        cands = self.index.search(question, k_bm25=30, k_faiss=30, k_out=25)
        top = self.reranker.rerank(question, cands, top_k=top_k_ctx)
        context = build_context(top)
        prompt = ANSWER_USER_PROMPT_TEMPLATE.format(question=question, context=context)
        raw = self.llm.generate(ANSWER_SYSTEM_PROMPT, prompt, max_new_tokens=200)
        # Try to parse JSON
        used_ids: List[int] = []
        answer_text = ""
        try:
            obj = json.loads(extract_json(raw))
            answer_text = obj.get("answer", "").strip()
            used_ids = [int(x) for x in obj.get("used_chunks", [])]
        except Exception:
            # Fallback: use top-3 IDs
            answer_text = raw.strip()
            used_ids = [c.id for c in top[:3]]
        citations = unique_citations(self.chunks, used_ids)
        return {"question": question, "answer": answer_text, "citations": citations}