# README.md

## RAG Prototype (word‑aware, table‑aware) for Russian Normative DOCX

This repo contains a single runnable script **`RAG_prototype_wordaware.py`** that builds a compact RAG system for Russian normative documents (.docx).

**Key ideas**
- **Word‑aware, sentence‑based chunking** (no mid‑word breaks) + **table‑aware** row chunks with table header as context.
- **Hybrid retrieval**: BM25 (lexical, RU) + FAISS (dense, `intfloat/multilingual-e5-large-instruct`) merged via **RRF**.
- **Cross‑encoder re‑ranking**: `jinaai/jina-reranker-v2-base-multilingual`.
- **Open‑source LLM backends**: **Ollama** (`Qwen2.5:7b`) or **HuggingFace local** (`Qwen/Qwen2.5-7B-Instruct`), deterministic (`temperature=0`).
- **Citations with locators**: document short name + auto‑extracted *пункт/таблица* in the final JSON.

---

## 1) Setup

### Python
- Python **3.10+** recommended.
- Install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Torch note**: If `torch` fails from `requirements.txt`, install it following the official command for your platform/GPU (see pytorch.org) and then re‑run `pip install -r requirements.txt`.

### Models
- **Ollama** backend (CPU/GPU):
  ```bash
  # Install Ollama from https://ollama.com
  ollama pull qwen2.5:7b
  ```
- **HF local** backend (GPU recommended):
  - Default: `Qwen/Qwen2.5-7B-Instruct` (bfloat16, device_map=auto).
  - On Colab or low VRAM: use 4‑bit quantization (`--backend hf --device cuda`, script auto‑enables 4‑bit for HF when `--device cuda`).

---

## 2) Data
Place all `.docx` normative files into a folder, e.g. `./docs/`.
Typical examples (your actual filenames may differ):
- `СП 91.13330.2012 Подземные горные выработки...docx`
- `Постановление_Правительства_РФ_от_16_02_2008_N_87_О_составе_разделов.docx`
- `СП 20.13330.2016 Нагрузки и воздействия...docx`
- `СП 31-115-2006 Открытые плоскостные физкультурно-спортивные сооружения.docx`
- `Приказ-Ростехнадзора-...N-505-...docx`

The loader ingests **paragraphs and tables**; tables are chunked by **rows** and enriched with the table header.

---

## 3) Run

### A) Local with Ollama
```bash
python RAG_prototype_wordaware.py --data_dir ./docs --backend ollama --model Qwen2.5:7b
```

### B) Local/Colab with HF (GPU)
```bash
python RAG_prototype_wordaware.py --data_dir ./docs --backend hf --model Qwen/Qwen2.5-7B-Instruct --device cuda
```

### Questions
- By default, the script answers the **10 assignment questions** embedded in the CLI.
- Alternatively, pass a custom file with one question per line:

```bash
python RAG_prototype_wordaware.py --data_dir ./docs --backend ollama --model Qwen2.5:7b --questions_file ./questions.txt
```

Results are printed to stdout and saved as `results_YYYYMMDD_HHMMSS.json`.

---

## 4) How it works (brief)
1. **Load & chunk**: sentence‑based windows (~1600 chars, 1‑sentence overlap); tables → header + row chunks. Extract **locators** (`п. 6.1.10`, `таблица 4.1`).
2. **Index**: BM25 over razdel tokens; FAISS (Inner Product) over **e5‑instruct** normalized embeddings.
3. **Merge**: **RRF** of BM25/FAISS candidates.
4. **Re‑rank**: cross‑encoder `jinaai/jina-reranker-v2-base-multilingual` to top‑K context.
5. **Generate**: LLM returns **JSON** `{ "answer": str, "used_chunks": [ids...] }` (deterministic). Script maps chunk ids → **citations** `{document, clause?, table?}`.

---

## 5) Output format
Each entry in `results_*.json` has:
```json
{
  "question": "...",
  "answer": "Краткий ответ с указанием пунктов/таблиц при наличии.",
  "citations": [
    {"document": "СП 91.13330.2012", "clause": "п. 6.1.10"},
    {"document": "СП 20.13330.2016", "table": "таблица 4.1"}
  ],
  "latency_sec": 1.234
}
```

---

## 6) Tuning tips
- Reduce/Increase sentence window (`max_chars`) to trade recall vs specificity.
- Adjust retrieval fan‑out: `k_bm25`, `k_faiss`, `k_out`, and reranker `top_k`.
- Swap embedding model to `ai-forever/sbert_large_nlu_ru` for RU‑heavy corpora and compare.
- If GPU is scarce, set backend to Ollama CPU and keep `top_k_ctx` small (6–8).

---

## 7) Troubleshooting
- **torch install** errors → install per platform from pytorch.org.
- **FAISS** import error on Apple Silicon → ensure `faiss-cpu` is used (not `faiss-gpu`).
- **Ollama** connection error → start daemon and `ollama pull qwen2.5:7b`.
- **Unicode in docx tables** → loader preserves punctuation/digits; only whitespace is normalized.

---

## 8) License / Models
All referenced models are open‑source. Check each model card for its specific license before distribution.
