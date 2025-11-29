# FLOWCHART_DETAILED.md

This document expands the flowchart into step-by-step operations for technical reviewers,
highlighting key decisions, failure modes, and recommended improvements for production.

## 1. Upload PDF
- User uploads a PDF via Streamlit file uploader.
- Accept only `application/pdf` types and validate file size (recommend < 50MB for web UI).

## 2. Extract Text with PyMuPDF (fitz)
- Use `fitz.open(stream=file_bytes, filetype='pdf')`.
- For each page call `page.get_text('text')`.
- Failure modes:
  - If text is missing (scanned PDF), detect and route to OCR (Tesseract).
  - If extraction throws errors (corrupt PDF), return a graceful error to the user.

## 3. Structure Text into Sections
- Heuristic rules:
  - Lines in UPPERCASE or matching headings regex become section headings.
  - Aggregate subsequent lines into that section.
- Recommendations:
  - Use layout/blocks from `get_text('blocks')` to preserve visual headings.
  - Merge very short sections into neighbors to avoid fragmentation.

## 4. Optional: Chunk & Embed Sections -> FAISS Index
- When documents are large, chunk into ~500-token chunks with overlap.
- Embed with a sentence-transformer and store in FAISS for semantic retrieval.
- At query time, embed the question and retrieve top-k chunks.

## 5. Model Loading & Caching
- Use `transformers` `AutoTokenizer` and `AutoModelForCausalLM`.
- Prefer `local_files_only=True` where desired to avoid network fetch.
- Streamlit: cache pipeline with `@st.cache_resource` to reuse model across requests.

## 6. Prompt Construction & RAG
- Prefer retrieval-augmented-generation:
  - Use only top-k relevant chunks in the prompt to stay under token limits.
- Include explicit instruction: "Answer concisely and cite section headings where applicable."

## 7. Inference & Postprocessing
- Use `max_new_tokens` to limit generation; avoid `max_length` pitfalls.
- Sanitize output: trim prompt echoes safely, strip control characters, and limit answer length.

## 8. Follow-up Questions
- Ask the model to produce 3 concise follow-ups.
- Postprocess: remove enumerators and ensure question-like phrasing (end with `?`).

## 9. Observability & Metrics
- Log: upload sizes, extraction success, retrieval latencies, model latency, errors.
- Add request IDs in logs for tracing.

## 10. Security & Secrets
- Never commit `.env` or tokens. Use `.env.example` and GitHub Secrets for CI.
- Validate user inputs and limit file sizes to mitigate DoS.

## 11. Deployment Notes
- Containerize with provided Dockerfile.
- Scale model inference separately (dedicated GPU node or serverless model endpoint).

