import streamlit as st
import tempfile
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

key = st.secrets['key']

llm = OpenAI(temperature=0, openai_api_key=key)


def summarize_pdf(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary


# side bar

with st.sidebar:
    doc_input = st.file_uploader(
        f"Upload your PDF", ["pdf"], accept_multiple_files=False
    )

    if st.button("Process"):
        with st.spinner("Processing"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(doc_input.getvalue())
                tmp_file_path = tmp_file.name
            summarize = summarize_pdf(tmp_file_path)


st.title("MEDICO BOT💉🩺")

st.header("The summary of the document is:")
try:
    st.markdown(summarize)
except:
    st.warning('Please upload your PDF',icon='🤖')