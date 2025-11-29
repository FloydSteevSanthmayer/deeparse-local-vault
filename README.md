# deeparse-local-vault

**Local-first PDF Question & Answer (Q&A) — DeepParse**

A professional, production-ready Streamlit application that enables question answering over PDF documents using the `deepparse/gpt-deepparse` model. The application downloads model weights **once** to the Hugging Face cache and runs inference locally, supporting offline/air-gapped deployments, optional retrieval-augmented generation (RAG) with FAISS, and OCR fallback for scanned documents.

---

## Key Features

- **Local-first inference** — model weights are downloaded once to the Hugging Face cache and reused on subsequent runs.
- **Streamlit UI** — clean web interface for uploading PDFs, asking questions, and viewing answers.
- **Robust text extraction** — PyMuPDF (`fitz`) for selectable text; optional OCR (Tesseract) fallback for scanned PDFs.
- **Sectioning & prompt controls** — heuristic section detection and the ability to select which sections to include in prompts.
- **Retrieval support (optional)** — chunk + embed workflow with `sentence-transformers` + `faiss` for semantic retrieval (RAG) on large documents.
- **Configurable generation** — adjust `max_new_tokens`, `temperature`, `top_k`, `top_p`, and sampling from the UI.
- **Docker-ready** — containerized for reproducible deployments.
- **CI & code quality** — GitHub Actions CI, Dependabot, and pre-commit hooks included.

---

## Quick Start

### Prerequisites
- Python 3.10+ (3.11 recommended)
- Git
- (Optional) CUDA-enabled GPU for faster inference
- Disk space for model weights (varies by model, often several hundred MB to multiple GB)

### Install & run locally
1. Clone the repository:
```bash
git clone https://github.com/<your-username>/deeparse-local-vault.git
cd deeparse-local-vault
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the example environment file and edit as needed:
```bash
cp .env.example .env
# Optionally set MODEL_ID, HF_TOKEN, DEVICE, MAX_NEW_TOKENS
```

5. Run the Streamlit app:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Run with Docker
```bash
docker build -t deeparse-local-vault .
docker run --rm -p 8501:8501 --env-file .env deeparse-local-vault
```
Visit `http://localhost:8501`.

---

## Model download behavior & offline mode

- On first run, `transformers` will download model tokenizer and weights to the Hugging Face cache (default: `~/.cache/huggingface/transformers`). This happens **only once** per machine/user.
- To enforce local-only operation (no network downloads), set `LOCAL_FILES_ONLY=true` in your `.env` and ensure the model artifacts are present in the cache. Alternatively, pre-download with:
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="deepparse/gpt-deepparse", cache_dir="/path/to/hf_cache", token="YOUR_TOKEN_IF_PRIVATE")
```

---

## Security & best practices

- **Do not commit secrets** — never commit `.env` or tokens; use GitHub Secrets for CI.
- Limit uploaded file sizes and validate content to avoid DoS.
- Consider isolating model inference on a dedicated node (GPU) for performance and access control.
- For air-gapped deployments, pre-populate the Hugging Face cache and run with `LOCAL_FILES_ONLY=true`.

---

## Project layout

```
.
├─ app.py
├─ README.md
├─ FLOWCHART_DETAILED.md
├─ flowchart_colored.mmd
├─ architecture.mmd
├─ docs/
│  ├─ flowchart_colored.png
│  └─ architecture.png
├─ Dockerfile
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ .github/
├─ tests/
├─ LICENSE
└─ CONTRIBUTING.md
```

---

## Contributing

Contributions are welcome. Please follow the developer guidelines in `CONTRIBUTING.md`:
- Fork the repository, create a feature branch, run tests locally, and open a PR against `main`.
- Keep changes small and include tests for new behavior.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Maintainer

Floyd Steev Santhmayer — *developer & maintainer*

For questions or support, open an issue in this repository.
