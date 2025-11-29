#!/usr/bin/env python3
"""Streamlit launcher for PDF Q&A (starter).

Note: This is a clean, production-minded entrypoint. It intentionally keeps model-loading
details in a helper to make testing easier.
"""

import os
import re
import fitz
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

load_dotenv()  # optional .env

MODEL_ID = os.environ.get("MODEL_ID", "deepparse/gpt-deepparse")
DEVICE   = int(os.environ.get("DEVICE", "-1"))  # -1 => CPU
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))

st.set_page_config(page_title="PDF Q&A (Local DeepParse)", layout="wide")
st.title("📄 PDF Q&A — Local DeepParse (Starter)")

@st.cache_resource
def load_pipeline(model_id: str, device: int):
    # Try local_files_only to avoid accidental downloads when desired; fall back to normal if not found.
    tokenizer = None
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
    except Exception:
        # fallback to network retrieval (may download on first run)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, return_full_text=False)
    return pipe

generator = load_pipeline(MODEL_ID, DEVICE)

def extract_text_from_pdf(file_bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        return "\n".join(pages).strip()
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""

def structure_pdf_content(raw_text: str) -> dict:
    sections = {"Introduction": []}
    current = "Introduction"
    heading_pattern = re.compile(r"^(?:CHAPTER|SECTION|PART|INTRODUCTION|CONCLUSION|SUMMARY|APPENDIX)\b", re.IGNORECASE)
    for line in raw_text.splitlines():
        text = line.strip()
        if not text:
            continue
        if text.isupper() or heading_pattern.match(text):
            current = text
            sections.setdefault(current, [])
        else:
            sections[current].append(text)
    return {sec: " ".join(lines) for sec, lines in sections.items()}

def call_model(prompt: str) -> str:
    out = generator(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[0]["generated_text"]
    if out.startswith(prompt):
        return out[len(prompt):].strip()
    return out.strip()

def build_prompt_from_sections(sections: dict, question: str, max_chars=40000):
    content = "\n".join(f"{k}: {v}" for k, v in sections.items())
    if len(content) > max_chars:
        # simple cutoff — recommended: replace with retrieval in production
        content = content[:max_chars]
    return f"Based on the following document content, answer the question:\n{question}\n\n{content}\nAnswer:"

uploaded = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded:
    raw_text = extract_text_from_pdf(uploaded.read())
    if raw_text:
        sections = structure_pdf_content(raw_text)
        st.success("PDF processed successfully — sections detected:\n" + ", ".join(list(sections.keys())[:10]))
        question = st.text_input("Ask a question about the document:")
        if question:
            with st.spinner("Generating answer..."):
                prompt = build_prompt_from_sections(sections, question)
                answer = call_model(prompt)
            st.subheader("Answer")
            st.write(answer)
            with st.expander("Follow-up Questions"):
                follow_prompt = f"Based on the following answer, generate 3 concise follow-up questions:\n{answer}\nQuestions:"
                raw = call_model(follow_prompt)
                lines = [re.sub(r"^\s*\d+\.\s*", "", l.strip()) for l in raw.splitlines() if l.strip()]
                for i, q in enumerate(lines[:3], 1):
                    st.write(f"{i}. {q}")
    else:
        st.error("No extractable text — the PDF may be scanned. Consider using OCR.")
else:
    st.info("Please upload a PDF to get started.")
