import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="wide")
st.title("📄 Local PDF RAG Assistant")

with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []

@st.cache_resource 
def process_pdf_and_create_chain(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    os.remove(temp_file_path)
    
    llm = ChatOllama(
        model="llama3.2",
        temperature=0.3,
        num_ctx=4096
    )
    
    prompt_template = """You are a helpful assistant. Answer the question using only the following context.
    If you cannot find the answer in the context, say "I don't know based on the provided information."

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

if uploaded_file is None:
    st.info("👈 Please upload a PDF file in the sidebar to get started.")
else:
    file_bytes = uploaded_file.read()
    
    with st.spinner("Processing PDF and building vector store..."):
        qa_chain = process_pdf_and_create_chain(file_bytes)
        
    st.success("PDF processed successfully! You can now ask questions.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                sources = response["source_documents"]
                
                st.markdown(answer)
                
                with st.expander("View Source Pages"):
                    for i, doc in enumerate(sources):
                        page_num = doc.metadata.get('page', 'Unknown')
                        st.info(f"**Source {i+1} (Page {page_num}):**\n\n{doc.page_content}")

        st.session_state.messages.append({"role": "assistant", "content": answer})