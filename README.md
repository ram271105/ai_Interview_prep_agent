# 🎯 AI Interview Coach

> An intelligent, resume-driven interview preparation platform powered by LangChain Agents and local LLMs via Ollama.
An AI-powered mock interview platform that reads your resume, generates personalized questions, evaluates your answers, and gives detailed feedback with scores — all running locally using Ollama and LangChain.

## 📌 Overview

AI Interview Coach is an end-to-end interview preparation system that:
- Parses your resume and understands your skills, projects, and experience
- Generates **personalized interview questions** based on your profile
- Evaluates your answers across multiple parameters
- Provides **detailed feedback, scores, and improvement suggestions**
- Exports a comprehensive **PDF performance report**

No generic question banks. No one-size-fits-all. Just questions built around **you**.

---

## 🏗️ Architecture

```
Resume Upload (PDF)
        │
        ▼
┌─────────────────┐
│  Resume Parser  │  ← Extracts skills, experience, projects
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐        ┌──────────────┐
│  LangChain Orchestrator  │◄──────►│  Ollama LLM  │
│  (Agent Controller)      │        │ (llama3/mistral)│
└────────┬─────────────────┘        └──────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌──────────┐ ┌─────────────────┐
│ Question │ │   Evaluation    │
│  Gen     │ │     Agent       │
│  Agent   │ │ (0-10 scoring)  │
└────┬─────┘ └────────┬────────┘
     │                │
     ▼                ▼
┌──────────────────────────┐
│   Report Generation      │
│  Charts + PDF Export     │
└──────────────────────────┘
```

---

## ✨ Features

- 📄 **Resume Parsing** — Extracts skills, projects, education, and work experience from PDF
- 🤖 **AI Question Generation** — Generates role-specific, resume-tailored interview questions
- 📝 **Answer Evaluation** — Scores answers on accuracy, clarity, relevance, understanding, and completeness
- 📊 **Performance Visualization** — Charts showing question-wise and skill-wise performance
- 📥 **PDF Report Export** — Downloadable interview report with full feedback
- 🔒 **Fully Local** — Runs on your machine via Ollama, no API keys needed

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM | Ollama (llama3 / mistral) |
| Orchestration | LangChain + LangGraph |
| Embeddings | HuggingFace sentence-transformers |
| Vector Store | Chroma / FAISS |
| Frontend | Streamlit |
| PDF Processing | PyMuPDF / pdfplumber |
| Report Export | ReportLab / FPDF |

---

## 📁 Folder Structure

```
ai-interview-coach/
├── src/
│   ├── agents/
│   │   ├── question_gen_agent.py
│   │   ├── evaluation_agent.py
│   │   └── orchestrator.py
│   ├── parser/
│   │   └── resume_parser.py
│   ├── chains/
│   │   └── langchain_chains.py
│   └── report/
│       └── report_generator.py
├── data/
│   └── sample_resumes/
├── notebooks/
│   └── experiments.ipynb
├── tests/
│   └── test_agents.py
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-interview-coach.git
cd ai-interview-coach
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and run Ollama
```bash
# Install Ollama from https://ollama.com
ollama pull llama3
```

### 5. Configure environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 6. Run the app
```bash
streamlit run app.py
```

---

## 🚀 Usage

1. Open the app in your browser at `http://localhost:8501`
2. Upload your resume in PDF format
3. Wait for the system to parse and analyze your profile
4. Answer the generated interview questions one by one
5. View your scores, feedback, and performance charts
6. Download your PDF report

---

## 📊 Evaluation Parameters

Each answer is scored on a scale of **0–10** across:

| Parameter | Description |
|---|---|
| Technical Accuracy | Correctness of the answer |
| Conceptual Understanding | Depth of knowledge shown |
| Relevance | How well it addresses the question |
| Communication Clarity | Structure and readability |
| Completeness | Coverage of all aspects |

---

## 📦 Requirements

```
langchain
langgraph
ollama
streamlit
chromadb
faiss-cpu
sentence-transformers
pdfplumber
pymupdf
reportlab
python-dotenv
```

---

##  Future Enhancements

- [ ] Fine-tune LLM on domain-specific interview data
- [ ] Add HR, technical, and behavioral round agents
- [ ] Integrate real-time job description matching
- [ ] Deploy as production-ready web service
- [ ] Track progress across multiple sessions




##  Acknowledgements

- [LangChain](https://docs.langchain.com) — Agent orchestration framework
- [LangGraph](https://langchain-ai.github.io/langgraph) — Stateful agent graphs
- [Ollama](https://ollama.com) — Local LLM inference
- [Chroma DB](https://docs.trychroma.com) — Vector database
- [Streamlit](https://docs.streamlit.io) — UI framework
- [HuggingFace](https://huggingface.co) — Embedding models
