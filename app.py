
from __future__ import annotations

import os
import re
import io
import logging
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF
import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# optional: try to enable OCR fallback if pytesseract & PIL are installed
_HAS_OCR = False
try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore

    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

# ----------------------------
# Configuration (env-friendly)
# ----------------------------
MODEL_ID = os.environ.get("MODEL_ID", "deepparse/gpt-deepparse")
# DEVICE: -1 => CPU, >=0 => CUDA device index
DEVICE = int(os.environ.get("DEVICE", "-1"))
# default tokens for generation (this controls *new* tokens)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # optional token for private HF repos
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "false").lower() in ("1", "true", "yes")

# prompt/content guards
MAX_PROMPT_CHARS = 60_000  # naive char limit to avoid exceeding model context window

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_qa_app")

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="PDF Q&A (Local DeepParse)", layout="wide")
st.title("📄 PDF Q&A with Local DeepParse (Updated)")

# ------------------------------------------------------------
# Load tokenizer & model once (cached by Streamlit)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model() -> pipeline:
    """
    Load tokenizer + model into a Hugging Face transformers pipeline.
    Tries local_files_only first if configured; otherwise falls back to network download.
    Returns a text-generation pipeline ready for inference.
    """
    logger.info("Loading model pipeline. model=%s device=%s local_only=%s", MODEL_ID, DEVICE, LOCAL_FILES_ONLY)
    tokenizer_kwargs = {}
    model_kwargs = {}
    if HF_TOKEN:
        tokenizer_kwargs["use_auth_token"] = HF_TOKEN
        model_kwargs["use_auth_token"] = HF_TOKEN

    # attempt local-only load first if requested
    if LOCAL_FILES_ONLY:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True, **tokenizer_kwargs)
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, local_files_only=True, **model_kwargs)
            gen_pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=DEVICE,
                return_full_text=False,
            )
            logger.info("Loaded model from local cache (local_files_only=True).")
            return gen_pipe
        except Exception as e:
            logger.warning("Local-only load failed: %s — falling back to online load. Error: %s", MODEL_ID, e)

    # default load (may download if not cached)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        return_full_text=False,
    )
    logger.info("Loaded model (from cache or network).")
    return gen_pipe


# instantiate (cached)
try:
    generator = load_model()
except Exception as e:
    # If model loading fails, surface a helpful message in the UI
    st.error("Failed to load model. Check MODEL_ID, network, or HF_TOKEN. See console for details.")
    logger.exception("Model load error: %s", e)
    # keep generator as None to avoid NameError later; calls should check
    generator = None  # type: ignore

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, bool]:
    """
    Extract text from uploaded PDF bytes.
    Returns (text, used_ocr_flag). If OCR is used, used_ocr_flag=True.
    """
    used_ocr = False
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = []
        for page in doc:
            # use 'text' extraction which is usually robust for selectable text
            pages_text.append(page.get_text("text") or "")
        raw = "\n".join(pages_text).strip()

        # If no text present (likely scanned PDF) and OCR is available, attempt OCR
        if not raw and _HAS_OCR:
            used_ocr = True
            ocr_pages = []
            for i in range(len(doc)):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                try:
                    page_text = pytesseract.image_to_string(img)
                except Exception as e:
                    logger.warning("pytesseract failed on page %s: %s", i, e)
                    page_text = ""
                ocr_pages.append(page_text or "")
            raw = "\n".join(ocr_pages).strip()
        return raw, used_ocr
    except Exception as e:
        logger.exception("Error extracting PDF text: %s", e)
        return "", used_ocr


def structure_pdf_content(raw_text: str) -> Dict[str, str]:
    """
    Split raw text into sections by detecting headings heuristically.
    Keeps your original heuristic but adds small improvements (length threshold).
    """
    sections: Dict[str, List[str]] = {"Introduction": []}
    current = "Introduction"
    heading_pattern = re.compile(r"^(?:CHAPTER|SECTION|PART|INTRODUCTION|CONCLUSION|SUMMARY|APPENDIX)\b", re.IGNORECASE)

    for line in raw_text.splitlines():
        text = line.strip()
        if not text:
            continue
        # Treat short-to-moderate-length UPPERCASE lines as headings (avoid giant paragraphs)
        if (len(text) <= 200 and text.isupper()) or heading_pattern.match(text):
            current = text
            sections.setdefault(current, [])
        else:
            sections[current].append(text)
    # join lists into single strings per section
    return {sec: " ".join(lines).strip() for sec, lines in sections.items()}


def call_model(prompt: str) -> str:
    """
    Generate a response from the local DeepParse model.
    Uses max_new_tokens to limit generation and attempts to strip echoed prompt safely.
    """
    if generator is None:
        logger.error("Generator pipeline is not available.")
        return "⚠️ Model not loaded. Check server logs."

    # ensure we do not pass an absurdly long prompt blindly
    if len(prompt) > MAX_PROMPT_CHARS:
        logger.warning("Prompt is large (chars=%d); truncated to %d chars.", len(prompt), MAX_PROMPT_CHARS)
        prompt = prompt[:MAX_PROMPT_CHARS] + "\n\n[TRUNCATED PROMPT]"

    try:
        # call pipeline with max_new_tokens (controls *new* tokens)
        out = generator(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        # pipeline returns list of dicts; safe access
        if isinstance(out, list) and len(out) > 0:
            generated_text = out[0].get("generated_text", "")
        elif isinstance(out, dict):
            generated_text = out.get("generated_text", "")
        else:
            generated_text = str(out)

        # If the model (for some pipelines) returns prompt+generation, remove prompt prefix safely.
        # Only remove if it's an exact prefix (avoid accidental truncation).
        if generated_text.startswith(prompt):
            return generated_text[len(prompt) :].strip()
        return generated_text.strip()
    except Exception as e:
        logger.exception("Model generation failed: %s", e)
        return f"⚠️ Model generation error: {e}"


def query_pdf_content(sections: Dict[str, str], question: str) -> str:
    """
    Build a simple prompt combining the sections and ask the model the question.
    For large documents you should replace this with retrieval (embedding + FAISS).
    """
    content = "\n".join(f"{sec}: {body}" for sec, body in sections.items())
    prompt = f"Based on the following document content, answer this question concisely and cite section headings when helpful:\n\nQuestion: {question}\n\nDocument:\n{content}\n\nAnswer:"
    return call_model(prompt)


def generate_followup_questions(answer: str) -> List[str]:
    prompt = f"Based on the following short answer, generate 3 concise follow-up questions that a human might ask for clarification:\n\nAnswer:\n{answer}\n\nQuestions (numbered):"
    raw = call_model(prompt)
    # remove numbering like "1. " and filter empty lines
    lines = [re.sub(r"^\s*\d+\.\s*", "", l.strip()) for l in raw.splitlines() if l.strip()]
    # dedupe and return up to 3 items
    seen = set()
    result = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            result.append(l)
        if len(result) >= 3:
            break
    return result or ["❌ No follow-up questions available."]


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.info(f"Model: `{MODEL_ID}` — Device: `{DEVICE}` — local-only: `{LOCAL_FILES_ONLY}`")

uploaded = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded:
    raw_bytes = uploaded.read()
    with st.spinner("Extracting text from PDF..."):
        raw_text, used_ocr = extract_text_from_pdf(raw_bytes)

    if not raw_text:
        st.error("Failed to extract text from the uploaded PDF. If the PDF is scanned, install OCR tools (pytesseract) or enable OCR.")
    else:
        sections = structure_pdf_content(raw_text)
        st.success("PDF processed successfully!" + (" (OCR used)" if used_ocr else ""))

        # Show a simple TOC and allow selection of sections to include in prompt
        st.subheader("Document Sections (table of contents)")
        toc = list(sections.keys())
        if len(toc) == 0:
            st.warning("No sections detected; the document will be used as a single block.")
        else:
            cols = st.columns(3)
            for i, sec in enumerate(toc):
                cols[i % 3].write(f"- {sec}")

        st.markdown("---")
        st.write("Tip: By default the app will include all sections in the model prompt. For large PDFs, select a subset of important sections using the multi-select below to keep prompts smaller and more focused.")
        selected = st.multiselect("Select sections to include in the prompt (optional)", options=toc, default=toc[:3] if toc else None)

        # question input
        question = st.text_input("Ask a question about the document:")

        if question:
            with st.spinner("Generating answer..."):
                # build content used in prompt
                if selected:
                    content = "\n".join(f"{sec}: {sections[sec]}" for sec in selected if sec in sections)
                else:
                    content = "\n".join(f"{sec}: {body}" for sec, body in sections.items())

                # guard prompt size
                if len(content) > MAX_PROMPT_CHARS:
                    # naive cutoff — production: use RAG (embeddings + retrieval)
                    content = content[:MAX_PROMPT_CHARS] + "\n\n[CONTENT TRUNCATED — consider enabling retrieval-based prompting]"

                prompt = f"Based on the following document content, answer this question concisely and cite section headings when helpful:\n\nDocument:\n{content}\n\nQuestion: {question}\n\nAnswer:"
                answer = call_model(prompt)

            st.subheader("Answer")
            st.write(answer)

            # Follow-up questions
            with st.expander("Follow-up Questions"):
                followups = generate_followup_questions(answer)
                for i, fq in enumerate(followups, 1):
                    st.write(f"{i}. {fq}")

            # Download button
            st.download_button("Download answer as .txt", data=answer.encode("utf-8"), file_name="answer.txt", mime="text/plain")

            # optional debug: show prompt
            with st.expander("Show prompt (debug)"):
                st.code(prompt[:15000] + ("\n\n... (truncated)" if len(prompt) > 15000 else ""), language="text")

else:
    st.info("Please upload a PDF to get started.")

# Footer with tips
st.markdown("---")
st.markdown(
)
