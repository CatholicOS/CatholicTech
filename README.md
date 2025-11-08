# Instrumentum Veritatis
*A CatholicOS Hackathon Project by CatholicTech*

## Overview
**Instrumentum Veritatis** is an AI model designed to serve as a responsible and faithful assistant for exploring the Catholic Church’s teachings on moral, social, and economic issues.  
It draws from official Church documents such as:
- Encyclicals and Apostolic Letters  
- Papal Bulls and Councils  
- Writings of the Church Fathers  
- The Catechism of the Catholic Church  

Our goal is not to replace theological reasoning, but to **assist human understanding** by organizing, summarizing, and providing accessible insights into the Church’s intellectual and moral tradition.

---

## How It Works
1. **Corpus Building** – We collect and structure Church documents (in `.txt` or `.pdf` format).  
2. **Text Embedding** – Each document is processed into vector embeddings for semantic search.  
3. **Question Answering** – A Large Language Model (LLM) retrieves relevant passages and provides an answer, citing sources transparently.  
4. **Interface** – A simple web or Streamlit app for asking questions and viewing cited results.

---

## Tech Stack
- **Python 3.12+**
- **OpenAI or local LLM (Llama/Mistral)**
- **LangChain / LlamaIndex** for retrieval
- **FAISS / ChromaDB** for semantic search
- **Streamlit** or **Flask** for the front-end

---

## Example Use
> “What does the Church teach about the dignity of work?”
>  
> → Cites *Laborem Exercens* (John Paul II), *Rerum Novarum* (Leo XIII), and relevant Catechism sections.

---

## Ethical Principle
> *“The human person, created in the image of God, is the measure of all technological progress.”*  
> — Pontifical Academy for Life, *Ethics in Artificial Intelligence* (2020)

This project aims to embody that principle by ensuring AI is used **to aid human wisdom**, not replace it.
