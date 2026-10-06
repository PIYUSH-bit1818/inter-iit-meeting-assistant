"""Streamlit UI. Run from repo root: streamlit run frontend/app.py"""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Meeting Assistant", layout="wide")
st.title("Meeting Assistant")
st.caption("Speech-to-text → domain-aware refinement → meeting minutes, decisions and action items")

with st.sidebar:
    st.subheader("Backend")
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        resp.raise_for_status()
        health = resp.json()
        st.success(f"Connected ({BACKEND_URL})")
        for key, ok in health.get("config", {}).items():
            st.write(("✅ " if ok else "⚠️ ") + key)
    except requests.RequestException as exc:
        st.error(f"Backend unreachable at {BACKEND_URL}: {exc.__class__.__name__}")

uploaded = st.file_uploader(
    "Upload an English meeting recording",
    type=["wav", "mp3", "m4a", "ogg", "flac", "webm", "mp4"],
)

st.button("Process recording", disabled=True, help="Processing is enabled in a later phase.")
if uploaded is not None:
    st.info("Processing pipeline not implemented yet.")
