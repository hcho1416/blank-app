pip install streamlit google-genai
import streamlit as st
from google import genai
from google.genai import types

# 1. Page Configuration and UI Header
st.set_page_config(
    page_title="Academic Transcript Evaluator",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Post-Baccalaureate Transcript Evaluation Tool")
st.markdown("""
This application assists academic advisors by cross-referencing completed student transcripts 
against mandatory program advising sheets for teacher licensure.
""")
st.markdown("---")

# 2. Configuration Sidebar (API and Model Selection)
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Enter Gemini API Key:", 
        type="password", 
        help="Provide your developer API key to run evaluations securely."
    )
    model_choice = st.selectbox(
        "Select Model Engine:",
        ["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        help="Use 'flash' for fast results or 'pro' for advanced, nuanced credit evaluations."
    )
    
    st.info(
        "🔒 **FERPA & Privacy Notice:** Files are processed via inline memory buffers "
        "and sent to your private API context. Ensure your API agreement aligns with institutional data policy."
    )

# 3. File Upload Architecture
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Official Student Transcript")
    transcript_file = st.file_uploader(
        "Upload Transcript (PDF)", 
        type=["pdf"], 
        key="transcript_upload"
    )

with col2:
    st.subheader("2. Program Advising Sheet")
    advising_file = st.file_uploader(
        "Upload Advising Sheet / Requirements (PDF)", 
        type=["pdf"], 
        key="advising_upload"
    )

# 4. Processing and Evaluation Logic
if st.button("Run Transcript Evaluation", type="primary"):
    if not api_key:
        st.error("⚠️ Evaluation halted: Please enter your Gemini API Key in the sidebar.")
    elif not transcript_file or not advising_file:
        st.error("⚠️ Evaluation halted: Both a transcript and an advising sheet must be uploaded to continue.")
    else:
        with st.spinner("AI Engine is executing cross-reference checks... (This may take up to 30 seconds)"):
            try:
                # Initialize the Google GenAI Client
                client = genai.Client(api_key=api_key)
                
                # Convert the Streamlit UploadedFile objects directly into raw byte parts
                transcript_bytes = transcript_file.read()
                advising_bytes = advising_file.read()
                
                transcript_part = types.Part.from_bytes(
                    data=transcript_bytes,
                    mime_type="application/pdf"
                )
                advising_part = types.Part.from_bytes(
                    data=advising_bytes,
                    mime_type="application/pdf"
                )
                
                # 5. Rigorous Prompts for Curricular Equivalence
                evaluation_prompt = """
                You are an expert university registrar and academic advising assistant evaluating post-baccalaureate transcripts for teaching licensure.
                
                Analyze the two attached PDF documents:
                1. The Student Transcript (historical coursework, grades, degrees).
                2. The Advising Sheet / Licensure Requirements (required competencies, courses, benchmarks).
                
                Your task is to comprehensively cross-reference every required competency or course block on the advising sheet against the student's completed transcript coursework. Look for exact matches by course number or close content equivalencies (e.g., "Intro to Calculus" satisfying a foundational math requirement, or "General Psychology" mapping to human development modules).
                
                Output your evaluation strictly as a clean Markdown table with the following structured columns:
                - **Licensure Requirement**: The required course code, title, or competency block from the advising sheet.
                - **Status**: Mark clearly as [Satisfied], [Missing], or [Needs Faculty Review].
                - **Matched Transcript Course**: The exact course code, title, term completed, and final grade from the transcript. (Leave blank if missing).
                - **Credits Applied**: Number of credits accepted / number required.
                - **Advisor Notes**: Academic justification for the matching logic or specific instructions regarding missing benchmarks.
                
                CRITICAL CONSTRAINTS:
                - Do not assume equivalency if content fields diverge dramatically. If a match is highly borderline, mark it as 'Needs Faculty Review'.
                - Ensure failed courses (F, W, I grades) are never marked as 'Satisfied'.
                - Provide absolute fidelity to the source documents; do not hallucinate courses or requirements.
                """
                
                # Execute the API payload
                response = client.models.generate_content(
                    model=model_choice,
                    contents=[transcript_part, advising_part, evaluation_prompt]
                )
                
                # 6. Render Output Report
                st.success("✅ Analysis completed successfully!")
                st.subheader("📋 Academic Course Evaluation Report")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"An unexpected system exception occurred: {e}")
                st.info("Check if your API key is correct and valid for the selected model architecture.")
