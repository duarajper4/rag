# 🎬 VidMind AI — YouTube Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-0064FF?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**An AI-powered Retrieval-Augmented Generation (RAG) system that lets you chat with any YouTube video.**

[Demo](#demo) · [Features](#features) · [Installation](#installation) · [Usage](#usage) · [Tech Stack](#tech-stack)

</div>

---

## 📌 Overview

**VidMind AI** transforms any YouTube video into an interactive knowledge base. Paste a YouTube URL, and the system automatically fetches the transcript, builds a semantic vector index, and lets you ask natural language questions — all powered by Groq's ultra-fast LLaMA 3.3 70B model.

> Built as a full-stack AI application demonstrating RAG architecture, vector databases, and LLM integration.

---

## ✨ Features

- 🎥 **YouTube Transcript Extraction** — Automatically fetches captions from any YouTube video
- 🧠 **RAG Pipeline** — Retrieval-Augmented Generation for accurate, context-grounded answers
- 🔍 **Semantic Search** — FAISS vector database with MiniLM sentence embeddings
- ⚡ **Groq LLaMA 3.3 70B** — Ultra-fast inference with state-of-the-art language model
- 💬 **Persistent Chat** — Full conversation history with one-click clear
- 📊 **Live Analytics** — Real-time word count, chunk count, and session stats in sidebar
- 🎨 **Professional Dark UI** — Custom-designed interface with Streamlit

---

## 🏗️ Architecture

```
YouTube URL
     │
     ▼
┌─────────────────────┐
│  Transcript Fetcher │  ← youtube-transcript-api
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   Text Splitter     │  ← LangChain RecursiveCharacterTextSplitter
│   (500 char chunks) │     chunk_size=500, chunk_overlap=50
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Embedding Model    │  ← SentenceTransformer (all-MiniLM-L6-v2)
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   FAISS Vector DB   │  ← IndexFlatL2 similarity search
└─────────────────────┘
     │
  [User Query]
     │
     ▼
┌─────────────────────┐
│  Context Retrieval  │  ← Top-K semantic search (k=3)
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   Groq LLaMA 3.3   │  ← llama-3.3-70b-versatile
│   Answer Generator  │
└─────────────────────┘
     │
     ▼
   Response
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/youtube-rag-system.git
cd youtube-rag-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up API Key

Create a `.streamlit/secrets.toml` file:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Get your free Groq API key at → [console.groq.com](https://console.groq.com)

### 4. Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 📦 Requirements

```txt
streamlit
youtube-transcript-api
sentence-transformers
faiss-cpu
groq
langchain-text-splitters
numpy
```

---

## 🎯 Usage

1. **Paste** any YouTube URL into the input field
2. Click **Transcribe & Process Video**
3. Wait for the transcript to be fetched and embeddings to be built
4. Switch to the **Chat with Video** tab
5. **Ask anything** about the video content
6. Use **🗑️ Clear** to reset the chat history anytime

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM** | Groq — LLaMA 3.3 70B Versatile |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Text Splitting** | LangChain Text Splitters |
| **Transcript API** | youtube-transcript-api |
| **Language** | Python 3.9+ |

---

## 📁 Project Structure

```
youtube-rag-system/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
│
└── .streamlit/
    └── secrets.toml        # API keys (local only, not pushed)
```

---

## ☁️ Deployment

### Deploy on Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set **Main file path** to `app.py`
5. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
6. Click **Deploy** 🚀

---

## ⚠️ Limitations

- Only works with YouTube videos that have **captions/subtitles enabled**
- Answers are based solely on the **video transcript** (no external knowledge)
- Very long videos may take longer to process during embedding generation

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Author

**Duaa Rajper**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/YOUR_LINKEDIN)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/YOUR_USERNAME)

---

<div align="center">
  <sub>Built with ❤️ using Streamlit, Groq, and FAISS</sub>
</div>
