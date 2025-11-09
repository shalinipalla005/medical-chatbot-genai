
#  Medical Chatbot using Generative AI (RAG-based)

This project implements an **AI-powered medical assistant** that can answer domain-specific medical questions based on knowledge extracted from a large collection of medical PDFs.  
It uses **Retrieval-Augmented Generation (RAG)** with **LangChain**, **Hugging Face embeddings**, **FAISS vector store**, and **Groq-hosted Llama models** to generate accurate, context-aware responses in real time.

---

##  Project Overview

This chatbot bridges the gap between **medical document search** and **AI question answering**.  
Instead of relying on the LLM’s limited memory, it retrieves relevant medical context dynamically and uses it to generate precise, evidence-based responses.

###  Core Workflow
1. **Document Loading:**  
   Loads and parses all PDFs from the `data/` directory.
2. **Text Chunking:**  
   Splits long documents into smaller overlapping chunks for efficient retrieval.
3. **Embedding Generation:**  
   Converts chunks into dense vector representations using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Vector Storage (FAISS):**  
   Stores all embeddings locally in a FAISS vector database for fast semantic search.
5. **Retrieval-Augmented Generation (RAG):**  
   When a query is asked, the top relevant chunks are retrieved and passed to the **Groq Llama model** for answer generation.
6. **Interactive Frontend:**  
   A user-friendly **Streamlit** interface to chat with the AI assistant.

---

##  Tech Stack

| Component | Technology Used | Purpose |
|------------|----------------|----------|
| **Frontend** | Streamlit | Chat-based web interface |
| **Model** | `meta-llama/llama-4-maverick-17b-128e-instruct` via Groq API | Generative AI reasoning |
| **Vector Store** | FAISS | Fast similarity search for embeddings |
| **Embeddings** | Hugging Face (`all-MiniLM-L6-v2`) | Converts text to numerical vectors |
| **Orchestration** | LangChain | Connects retrieval, prompt, and model |
| **Environment Management** | Python-dotenv | Secure API key handling |
| **Data Source** | Local Medical PDFs | Knowledge base for the chatbot |

---

##  Features

 **Domain-Specific RAG Pipeline** – Uses context from your own medical PDFs.  
 **Groq API Integration** – Leverages high-speed inference of Llama models.  
 **Streamlit Chat UI** – Simple, real-time, user-friendly interface.  
 **Custom Prompting** – Enforces factual, context-grounded answers only.  
 **Local Vector Storage** – Enables fast offline retrieval (FAISS).  
 **Scalable Design** – Easily extendable with more data or larger models.

---

##  Project Structure

```

medical-chatbot/
│
├── .env                        # Contains GROQ_API_KEY (DO NOT UPLOAD)
├── .gitignore                  # Ignores sensitive & generated files
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── create_memory_for_llm.py     # Loads PDFs, creates embeddings, saves FAISS index
├── medibot.py      # Streamlit chat interface
│
├── vectorstore/                 # Saved FAISS vector database
├── data/                        # Folder for medical PDFs
└── ...

````

---

## ⚙️ Installation & Setup

###  Clone the repository

```bash
git clone https://github.com/shalinipalla005/medical-chatbot-genai.git
cd medical-chatbot-genai
````

###  Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # On Windows
# source venv/bin/activate   # On Mac/Linux
```

###  Install dependencies

```bash
pip install -r requirements.txt
```

###  Add your environment variables

Create a `.env` file in the project root and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

*(Make sure `.env` is listed in your `.gitignore`.)*

###  Prepare the vector database

Before chatting, you need to process your PDFs:

```bash
python create_memory_for_llm.py
```

This step:

* Loads PDFs from `/data`
* Splits them into chunks
* Creates embeddings
* Saves them in `/vectorstore`

You’ll see something like:

```
Loaded Groq key: ******
Length of PDF pages: 759
```

---

##  Run the Chatbot

Once embeddings are created, start the Streamlit app:

```bash
streamlit run app.py
```

Then open the displayed **localhost link** (e.g., `http://localhost:8501`) in your browser.

---

##  Example Queries

You can ask questions like:

```
- What are the symptoms of diabetes?
- How is hypertension diagnosed?
- What is the normal blood pressure range?
```

And the chatbot will:

* Retrieve relevant context from your PDFs
* Generate a clean, factual, and formatted medical response
* Display source document references

---

##  Output Example

**You:** What are the causes of asthma?
**AI Response:**

> Asthma is caused by a combination of genetic and environmental factors.
> Common triggers include allergens, air pollution, respiratory infections, and stress.
>
> **Source Docs:** [document_12.pdf, document_45.pdf]

---

##  requirements.txt (Sample)

If you don’t already have one, include:

```
streamlit
langchain
langchain-community
langchain-huggingface
langchain-groq
faiss-cpu
sentence-transformers
python-dotenv
torch
torchvision
```

---

##  Security Note

* Never push your `.env` file to GitHub.
* Always list `.env`, `/vectorstore/`, and `/data/` in `.gitignore`.
* If deploying, use environment variables on the host platform.

---

##  Optional: Deployment

You can deploy this chatbot on:

* **Streamlit Cloud**
* **Hugging Face Spaces**
* **Render / Railway / Vercel (with Streamlit)**

Before deploying:

* Add your environment variable (`GROQ_API_KEY`) in the platform settings.
* Ensure FAISS and model dependencies are installed.

---

##  Future Enhancements

* Add **voice-based query input** using SpeechRecognition.
* Include **multi-modal input** (e.g., images or reports).
* Fine-tune the embedding model for specific medical sub-domains.
* Deploy the chatbot publicly via Streamlit Cloud or Hugging Face Spaces.

---

##  Acknowledgements

* [LangChain](https://www.langchain.com/)
* [Groq API](https://groq.com/)
* [Hugging Face](https://huggingface.co/)
* [Streamlit](https://streamlit.io/)
* [FAISS](https://faiss.ai/)

---


## Setting Up Your Environment with Pipenv

## Prerequisite: Install Pipenv
Follow the official Pipenv installation guide to set up Pipenv on your system:  
[Install Pipenv Documentation](https://pipenv.pypa.io/en/latest/installation.html)

---

## Steps to Set Up the Environment

### Install Required Packages
Run the following commands in your terminal (assuming Pipenv is already installed):

```bash
pipenv install langchain langchain_community langchain_huggingface faiss-cpu pypdf
pipenv install huggingface_hub
pipenv install streamlit
