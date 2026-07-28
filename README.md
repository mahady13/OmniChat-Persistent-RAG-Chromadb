# 💬 OmniChat AI with Persistent RAG

An intelligent multi-model AI chatbot built with **Streamlit**, **LangChain**, **OpenRouter**, **ChromaDB**, and **Hugging Face Embeddings**. OmniChat AI supports persistent Retrieval-Augmented Generation (RAG), allowing users to chat with their own PDF documents while also functioning as a general-purpose AI assistant.

---

## 🚀 Features

- 🤖 Chat with multiple free LLMs through OpenRouter
- 📄 Persistent PDF-based RAG using ChromaDB
- 🔍 Semantic document search with Hugging Face Embeddings
- 💾 Automatic vector database persistence
- 💬 Conversation history support
- ⚡ Streamlit-based modern UI
- 🔄 Switch between multiple AI models instantly
- 🧹 Clear conversation button
- 📚 Works with or without uploaded PDFs

---

## 🏗️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Web Application |
| LangChain | LLM Orchestration |
| OpenRouter | AI Model Provider |
| ChromaDB | Persistent Vector Database |
| Hugging Face Embeddings | Text Embeddings |
| MiniLM-L6-v2 | Embedding Model |
| RecursiveCharacterTextSplitter | Document Chunking |
| PyPDFDirectoryLoader | PDF Loading |

---

## 🤖 Available Models

Users can choose among multiple free AI models including:

- Ling 3 Flash
- Google Gemma 4-26B
- Nvidia Nemotron 3 Ultra
- Nvidia Nemotron 3 Super
- Nvidia Nano Omni
- Cohere North Mini Code
- PoolSide Laguna S2.1
- PoolSide Laguna XS2.1
- OpenAI GPT-OSS-20B
- OpenRouter Free Router

---

## 📂 Project Structure

```
OmniChat-AI/
│
├── assets/                 # PDF documents
├── chromadb/               # Persistent Vector Database
├── app.py                  # Main Streamlit Application
├── .env                    # API Key
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/OmniChat-AI.git
cd OmniChat-AI
```

---

### 2. Create Virtual Environment

Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Create `.env`

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

---

### 5. Add PDFs (Optional)

Place your PDF files inside:

```
assets/
```

If no PDFs are found, OmniChat AI automatically works as a normal chatbot.

---

### 6. Run the Application

```bash
streamlit run app.py
```

---

## 🧠 How Persistent RAG Works

1. PDFs are loaded from the `assets` folder.
2. Documents are split into smaller chunks.
3. Each chunk is converted into vector embeddings.
4. Embeddings are stored in ChromaDB.
5. On future launches, the existing vector database is reused.
6. User questions retrieve the most relevant document chunks before sending them to the selected LLM.

This makes responses faster since embeddings do not need to be regenerated every time.

---

## 📖 Workflow

```text
PDF Files
     │
     ▼
Document Loader
     │
     ▼
Text Splitter
     │
     ▼
Embeddings (MiniLM)
     │
     ▼
ChromaDB
     │
     ▼
Retriever
     │
     ▼
Prompt + Context
     │
     ▼
OpenRouter LLM
     │
     ▼
Response
```

---

## 📦 Requirements

- Python 3.10+
- Streamlit
- LangChain
- LangChain OpenAI
- LangChain Chroma
- LangChain HuggingFace
- ChromaDB
- python-dotenv
- sentence-transformers

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| OPENROUTER_API_KEY | Your OpenRouter API Key |

---
## 💡 Future Improvements

- Image understanding
- Voice input
- Voice output
- Conversation export
- Multiple document collections
- Web search integration
- Streaming citations
- Memory support
- Authentication
- Dark/Light theme switch

---

## 👨‍💻 Developer

**Mohiuddin Mahady**

BSc in Computer Science & Engineering

Mymensingh Engineering College

Affiliated with the University of Dhaka

- GitHub: https://github.com/mahady13
- LinkedIn: https://www.linkedin.com/in/mohiuddin-mahady/

---

## ⭐ If you found this project useful

Please consider giving the repository a ⭐ on GitHub.

It motivates me to continue building more open-source AI projects.

---

## 📄 License

This project is licensed under the MIT License.

Feel free to use, modify, and contribute.