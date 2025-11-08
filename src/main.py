# main.py
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from retrieval import get_vectorstore

st.set_page_config(page_title="Instrumentum Veritatis", page_icon="✝️")

st.title("✝️ Instrumentum Veritatis")
st.write("An AI assistant for exploring official Catholic teaching on moral, social, and economic questions.")

# Initialize vector store
with st.spinner("Loading theological corpus..."):
    vectordb = get_vectorstore()

# Setup the LLM
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# User Input
query = st.text_input("Ask a question (e.g., 'What does the Church teach about private property?')")

if query:
    with st.spinner("Thinking prayerfully..."):
        result = qa_chain(query)
        st.subheader("💬 Answer")
        st.write(result["result"])

        st.subheader("📚 Sources")
        for doc in result["source_documents"]:
            st.markdown(f"**File:** `{doc.metadata['source']}`")
            st.write(doc.page_content[:500] + "...")