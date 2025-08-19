def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Directory with .docx files")
    ap.add_argument("--backend", type=str, default="ollama", choices=["ollama", "hf"], help="LLM backend")
    ap.add_argument("--model", type=str, default="Qwen2.5:7b", help="Model name (e.g., 'Qwen2.5:7b' for Ollama or HF repo for --backend hf)")
    ap.add_argument("--device", type=str, default=None, help="Device for reranker/HF ('cuda' or 'cpu')")
    ap.add_argument("--questions_file", type=str, default=None, help="Path to file with questions (one per line)")
    args = ap.parse_args()

    engine = RAGEngine(Path(args.data_dir), backend=args.backend, model=args.model, device=args.device)

    if args.questions_file and Path(args.questions_file).exists():
        questions = [q.strip() for q in Path(args.questions_file).read_text(encoding='utf-8').splitlines() if q.strip()]
    else:
        # Default: the 10 questions from the assignment
        questions = [
            "В каких зонах по весу снежного покрова находятся Херсон и Мелитополь?",
            "Какие регионы Российской Федерации имеют высотный коэффициент k_h, превышающий 2?",
            "Выведи рекомендуемые варианты конструктивного решения заземлителей для стержневых молниеприемников.",
            "Что означает аббревиатура 'ТС'?",
            "Что должна содержать Пояснительная записка в графической части?",
            "Сколько разделов должна содержать проектная документация согласно 87ому постановлению?",
            "Какая максимальная скорость движения подземных машин в выработках?",
            "Какая максимальная температура допускается в горных выработках?",
            "Какие допустимые значения по отклонению геометрических параметров сечения горных выработок?",
            "В каком пункте указана минимальная толщина защитного слоя бетона для арматуры при креплении стволов монолитной бетонной крепью?",
        ]

    results = []
    for q in questions:
        t0 = time.time()
        out = engine.answer_one(q)
        out["latency_sec"] = round(time.time() - t0, 3)
        results.append(out)
        print("\nQ:", q)
        print("A:", out["answer"])
        print("Источники:")
        for c in out["citations"]:
            print(" - ", ", ".join([f"{k}: {v}" for k, v in c.items()]))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"results_{ts}.json")
    Path(out_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()