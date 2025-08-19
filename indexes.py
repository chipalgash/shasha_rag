
class HybridIndex:
    def __init__(self, chunks: List[Chunk], embed_model: str = "intfloat/multilingual-e5-large-instruct", faiss_path: Optional[Path] = None):
        self.chunks = chunks
        self.embed_model_name = embed_model
        self.faiss_path = Path(faiss_path) if faiss_path else None

        # BM25
        tokenized = [[t.text for t in tokenize(c.text.lower())] for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

        # Embeddings
        self._embedder = SentenceTransformer(self.embed_model_name)
        self._embedder_max = 4096
        vectors = self._encode_passages([c.text for c in chunks])
        self.dim = vectors.shape[1]

        # FAISS (cosine via inner product on normalized vectors)
        self._faiss = faiss.IndexFlatIP(self.dim)
        self._faiss.add(vectors)
        if self.faiss_path:
            faiss.write_index(self._faiss, str(self.faiss_path))

    def _encode_passages(self, texts: List[str]) -> np.ndarray:
        batch = []
        embs = []
        for t in texts:
            # e5-instruct expects "passage: " prefix
            batch.append(f"passage: {t[:self._embedder_max]}")
            if len(batch) == 64:
                e = self._embedder.encode(batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
                embs.append(e)
                batch = []
        if batch:
            e = self._embedder.encode(batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            embs.append(e)
        return np.vstack(embs)

    def _encode_query(self, q: str) -> np.ndarray:
        return self._embedder.encode([f"query: {q}"], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]

    def search(self, query: str, k_bm25: int = 30, k_faiss: int = 30, k_out: int = 25) -> List[Chunk]:
        # BM25
        bm25_scores = self._bm25.get_scores([t.text for t in tokenize(query.lower())])
        top_bm_idx = np.argsort(bm25_scores)[::-1][:k_bm25]

        # FAISS
        qv = self._encode_query(query)
        D, I = self._faiss.search(qv.reshape(1, -1), k_faiss)
        faiss_scores = D[0]
        faiss_idx = I[0]

        # RRF merge
        runs = [list(top_bm_idx), list(faiss_idx)]
        rrf_scores: Dict[int, float] = {}
        K = 60
        for run in runs:
            for rank, idx in enumerate(run, 1):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (K + rank)
        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k_out]
        return [self.chunks[i] for i, _ in merged]