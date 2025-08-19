class Reranker:
    def __init__(self, model: str = "jinaai/jina-reranker-v2-base-multilingual", device: Optional[str] = None):
        # device: "cuda" | "cpu" | None (auto)
        self._ce = CrossEncoder(model, device=device)

    def rerank(self, query: str, docs: List[Chunk], top_k: int = 10) -> List[Chunk]:
        pairs = [[query, d.text] for d in docs]
        scores = self._ce.predict(pairs, batch_size=32, show_progress_bar=False)
        order = np.argsort(scores)[::-1][:top_k]
        return [docs[i] for i in order]