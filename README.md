# PDF Q&A — Local DeepParse Streamlit App

**Professional, production-ready starter repository** for a Streamlit-based PDF Q&A application powered by a locally-hosted DeepParse model (Hugging Face). This repo includes architecture diagrams, CI, Dockerfile, tests scaffold, and developer docs.

---

## Quick links
- App entrypoint: `app.py`
- Flowchart (Mermaid): `flowchart_colored.mmd`
- Architecture (Mermaid): `architecture.mmd`
- Detailed flow explanation: `FLOWCHART_DETAILED.md`
- Rendered diagrams: `docs/flowchart_colored.png`, `docs/architecture.png`

---

## Overview
This project:
- Extracts text from uploaded PDFs (PyMuPDF / `fitz`)  
- Structures the text into sections for better prompt design  
- Uses a local Hugging Face model (e.g. `deepparse/gpt-deepparse`) via `transformers.pipeline` for question answering and follow-up generation  
- Provides a Streamlit UI and Docker + CI for reproducible runs

> Note: model files are downloaded once to Hugging Face cache by `transformers` and reused on subsequent runs.

---

## Does the **local host link** (http://localhost:8501) need to be included?
Short answer: **No, it's not required in the repo**, but the README includes instructions for running the app locally and how to access it at `http://localhost:8501` or the container host port when using Docker.  
When deployed (e.g., on a server, cloud VM, or Streamlit Sharing), you should use the public URL provided by that host instead.

---

## Prerequisites
- Python 3.10+ (3.11 recommended)
- Git
- (Optional) GPU + CUDA for model acceleration
- Enough disk space for model weights (hundreds of MB to multi-GB)

---

## Quickstart — Local
1. Clone:
```bash
git clone <your-repo-url>
cd pdf_qa_repo
```
2. Create venv & install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and fill values (e.g. `MODEL_ID`, `HF_TOKEN` if private):
```bash
cp .env.example .env
```
4. Run:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Run in Docker
```bash
docker build -t pdf-qa-app .
docker run --rm -p 8501:8501 --env-file .env pdf-qa-app
```
Then visit `http://localhost:8501`.

---

## Development & CI
- Tests: `pytest`
- Pre-commit hooks: `pre-commit install`
- GitHub Actions CI included: `.github/workflows/ci.yml`

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
├─ .github/workflows/ci.yml
├─ .github/dependabot.yml
├─ .pre-commit-config.yaml
├─ tests/
│  └─ test_app.py
├─ LICENSE
└─ CONTRIBUTING.md
```

---

## License
MIT — see `LICENSE`.

---

If you want, I can:
- Push this repo structure to a GitHub repo for you (I will provide commands), or
- Expand CI to build and publish a container image, or
- Replace the placeholder diagram PNGs with exports from FigJam (if you provide the images).

