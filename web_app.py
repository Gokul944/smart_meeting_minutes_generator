import streamlit as st
import os
from app import process_meeting
from io import BytesIO
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Smart Meeting Minutes Generator",
    layout="centered"
)

st.title("📝 Smart Meeting Minutes Generator")
st.write("Upload meeting audio and get structured meeting minutes.")

# ---------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------
if "processed" not in st.session_state:
    st.session_state.processed = False

if "overall" not in st.session_state:
    st.session_state.overall = ""

if "speaker_summaries" not in st.session_state:
    st.session_state.speaker_summaries = {}

if "actions" not in st.session_state:
    st.session_state.actions = []

if "metadata" not in st.session_state:
    st.session_state.metadata = {}

if "minutes_text" not in st.session_state:
    st.session_state.minutes_text = ""

if "gemini_error" not in st.session_state:
    st.session_state.gemini_error = None

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "m4a"]
)

# Optional: LLM toggle and API key
use_gemini = st.checkbox(
    "Use Google Gemini LLM for minutes (converts dialogue into formal 'what happened' minutes)",
    value=False,
)
gemini_api_key = None
if use_gemini:
    gemini_api_key = st.text_input(
        "Gemini API Key (or set GEMINI_API_KEY in .env / environment)",
        type="password",
        placeholder="Paste your API key here, or leave blank if set in .env",
        help="Get a key at https://aistudio.google.com/apikey",
    )

# ---------------------------------------------------
# PROCESS BUTTON (BEST PRACTICE)
# ---------------------------------------------------
if uploaded_file is not None:

    if st.button("🚀 Process Meeting Audio"):

        os.makedirs("uploads", exist_ok=True)
        upload_path = os.path.join("uploads", "meeting.wav")

        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        progress = st.progress(0)
        status = st.empty()

        try:
            with st.spinner("Processing meeting audio..."):
                progress.progress(10)
                status.text("Analyzing audio...")

                overall, speaker_summaries, actions, metadata, minutes_text, gemini_error = process_meeting(
                    upload_path,
                    use_llm=use_gemini,
                    gemini_api_key=gemini_api_key if use_gemini else None,
                )

                # SAVE RESULTS TO SESSION STATE
                st.session_state.overall = overall
                st.session_state.speaker_summaries = speaker_summaries
                st.session_state.actions = actions
                st.session_state.metadata = metadata
                st.session_state.minutes_text = minutes_text
                st.session_state.gemini_error = gemini_error
                st.session_state.processed = True

                progress.progress(100)
                status.text("Processing completed")

            st.success("✅ Meeting minutes generated successfully!")
            if st.session_state.gemini_error:
                st.warning("⚠️ Gemini API failed (rule-based minutes used): " + st.session_state.gemini_error)

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ---------------------------------------------------
# DISPLAY RESULTS (ONLY IF PROCESSED)
# ---------------------------------------------------
if st.session_state.processed:

    # -------- METADATA --------
    st.subheader("📌 Metadata")
    for k, v in st.session_state.metadata.items():
        if isinstance(v, list):
            st.write(f"**{k}:** {', '.join(v) if v else 'Not found'}")
        else:
            st.write(f"**{k}:** {v}")

    # -------- STRUCTURED MINUTES --------
    st.subheader("🧾 Structured Meeting Minutes")
    st.text(st.session_state.minutes_text)

    # -------- SPEAKER-WISE SUMMARY (visible) --------
    st.subheader("👥 Speaker-wise Summary")
    for sp, txt in st.session_state.speaker_summaries.items():
        st.markdown(f"**{sp}:** {txt}")

    # ======================================================
    # DOWNLOAD SECTION
    # ======================================================
    st.subheader("⬇️ Download Meeting Minutes")

    # -------- TXT --------
    # Use the structured minutes text as the TXT content
    text_content = st.session_state.minutes_text or ""

    st.download_button(
        "📄 Download as TXT",
        text_content,
        file_name="meeting_minutes.txt"
    )

    # -------- DOCX --------
    doc = Document()
    doc.add_heading("Meeting Minutes", level=1)
    # Write the structured minutes text into the DOCX
    for line in (st.session_state.minutes_text or "").splitlines():
        doc.add_paragraph(line)

    doc_io = BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)

    st.download_button(
        "📝 Download as DOCX",
        doc_io,
        file_name="meeting_minutes.docx"
    )

    # -------- PDF --------
    pdf_io = BytesIO()
    styles = getSampleStyleSheet()
    pdf = SimpleDocTemplate(pdf_io)

    story = []
    story.append(Paragraph("<b>Meeting Minutes</b>", styles["Title"]))
    for line in (st.session_state.minutes_text or "").splitlines():
        # Use simple paragraphs per line
        if line.strip().startswith("- "):
            story.append(Paragraph(line, styles["Normal"]))
        elif line.strip().endswith(":"):
            story.append(Paragraph(f"<b>{line}</b>", styles["Heading2"]))
        else:
            story.append(Paragraph(line, styles["Normal"]))

    pdf.build(story)
    pdf_io.seek(0)

    st.download_button(
        "📑 Download as PDF",
        pdf_io,
        file_name="meeting_minutes.pdf"
    )