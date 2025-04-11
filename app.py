import streamlit as st
import os
import zipfile

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Unzip the FAISS index if needed
if not os.path.exists("process_safety_index"):
    import zipfile
    with zipfile.ZipFile("process_safety_index.zip", "r") as zip_ref:
        zip_ref.extractall(".")

# Dynamically search for folder containing 'index.faiss'
def find_faiss_index_folder(root_dir="."):
    for root, dirs, files in os.walk(root_dir):
        if "index.faiss" in files and "index.pkl" in files:
            return root
    raise ValueError("❌ Could not find FAISS index files after unzipping.")

# Detect the correct folder
faiss_folder = find_faiss_index_folder()

# Load vector DB
vector_db = FAISS.load_local(
    faiss_folder,
    OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
)


# Set up custom prompt
custom_prompt = """
You are a Process Safety Assistant helping users understand complex EHS and Process Safety Standards.

Users may use informal, incorrect, or alternative terms. Your job is to:
- Understand the **intent** behind the question
- Match them to correct safety concepts from the documents

Question: {question}
Answer:
"""

prompt_template = PromptTemplate(input_variables=["question"], template=custom_prompt)
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Define query function
def ask_with_sources(question: str):
    docs = vector_db.similarity_search(question, k=5)
    answer = llm_chain.run({"question": question})

    links = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page_number", 1)
        chunk = doc.metadata.get("chunk_index", 0)
        link = f"https://your-company.com/viewer?file={source}&page={page}&highlight=chunk_{chunk}"
        links.append(f"{source} (Page {page}) → [Link]({link})")

    return answer, links

# Streamlit UI
st.set_page_config(page_title="Process Safety Assistant", layout="centered")
st.title("🧠 Process Safety Assistant")
st.markdown("Ask anything about Process Safety and get standards-based answers.")

user_input = st.text_input("Ask a question (in any language):")

if user_input:
    answer, links = ask_with_sources(user_input)

    st.markdown("### 💬 Answer")
    st.write(answer)

    st.markdown("### 📎 Sources")
    for link in links:
        st.markdown(f"- {link}")
