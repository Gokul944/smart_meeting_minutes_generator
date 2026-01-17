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

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "m4a"]
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

                overall, speaker_summaries, actions, metadata = process_meeting(upload_path)

                # SAVE RESULTS TO SESSION STATE
                st.session_state.overall = overall
                st.session_state.speaker_summaries = speaker_summaries
                st.session_state.actions = actions
                st.session_state.metadata = metadata
                st.session_state.processed = True

                progress.progress(100)
                status.text("Processing completed")

            st.success("✅ Meeting minutes generated successfully!")

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

    # -------- OVERALL SUMMARY --------
    st.subheader("🧾 Overall Summary")
    st.write(st.session_state.overall)

    # -------- SPEAKER SUMMARY --------
    st.subheader("👥 Speaker-wise Summary")
    for sp, txt in st.session_state.speaker_summaries.items():
        st.markdown(f"**{sp}:** {txt}")

    # -------- ACTION ITEMS --------
    st.subheader("✅ Action Items")
    for a in st.session_state.actions:
        st.write(f"- {a}")

    # ======================================================
    # DOWNLOAD SECTION
    # ======================================================
    st.subheader("⬇️ Download Meeting Minutes")

    # -------- TXT --------
    text_content = f"""
MEETING METADATA
{st.session_state.metadata}

OVERALL SUMMARY
{st.session_state.overall}

SPEAKER SUMMARIES
{st.session_state.speaker_summaries}

ACTION ITEMS
{st.session_state.actions}
"""

    st.download_button(
        "📄 Download as TXT",
        text_content,
        file_name="meeting_minutes.txt"
    )

    # -------- DOCX --------
    doc = Document()
    doc.add_heading("Meeting Minutes", level=1)

    for k, v in st.session_state.metadata.items():
        doc.add_paragraph(f"{k}: {v}")

    doc.add_heading("Overall Summary", level=2)
    doc.add_paragraph(st.session_state.overall)

    doc.add_heading("Speaker-wise Summary", level=2)
    for sp, txt in st.session_state.speaker_summaries.items():
        doc.add_paragraph(f"{sp}: {txt}")

    doc.add_heading("Action Items", level=2)
    for a in st.session_state.actions:
        doc.add_paragraph(f"- {a}")

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
    story.append(Paragraph("<b>Overall Summary</b>", styles["Heading2"]))
    story.append(Paragraph(st.session_state.overall, styles["Normal"]))

    story.append(Paragraph("<b>Action Items</b>", styles["Heading2"]))
    for a in st.session_state.actions:
        story.append(Paragraph(f"- {a}", styles["Normal"]))

    pdf.build(story)
    pdf_io.seek(0)

    st.download_button(
        "📑 Download as PDF",
        pdf_io,
        file_name="meeting_minutes.pdf"
    )
