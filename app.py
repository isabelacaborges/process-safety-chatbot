import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS

# Load PDFs from the pdfs/ folder
def load_documents():
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for filename in os.listdir("pdfs"):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("pdfs", filename))
            raw_pages = loader.load()
            for doc in raw_pages:
                chunks = splitter.split_documents([doc])
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "source": filename,
                        "page_number": doc.metadata.get("page", 1),
                        "chunk_index": i
                    })
                    docs.append(chunk)
    return docs

# Load and embed documents
@st.cache_resource(show_spinner=True)
def embed_documents():
    docs = load_documents()
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.from_documents(docs, embeddings)

# Initialize vector DB
st.title("🧠 Process Safety Assistant")
st.markdown("Ask about any standard or procedure you've uploaded.")
st.divider()

vector_db = embed_documents()

# Set up custom prompt
template = """
You are a Process Safety Assistant helping users understand complex EHS and Process Safety Standards.

Users may use informal, incorrect, or alternative terms. Your job is to:
- Understand the **intent** behind the question
- Match them to correct safety concepts from the documents

Question: {question}
Answer:
"""

prompt = PromptTemplate(input_variables=["question"], template=template)
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Chat UI
user_input = st.text_input("Ask a question about Process Safety:")

if user_input:
    with st.spinner("Searching your documents..."):
        answer = llm_chain.run({"question": user_input})
        sources = vector_db.similarity_search(user_input, k=3)

    st.markdown("### 💬 Answer")
    st.write(answer)

    st.markdown("### 📎 Sources")
    for src in sources:
        page = src.metadata.get("page_number", 1)
        file = src.metadata.get("source", "PDF")
        st.markdown(f"- {file} (Page {page})")
