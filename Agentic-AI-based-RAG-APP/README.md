# AI Tutor Platform
*A LLM powered application with deep/brief answers, citations, grading, and quizzes.*

---
## System Architecture

![AI Tutor Architecture](architecture.png)

## Overview

This project is an **AI tutoring system** built with:

- **Flask** – Web server  
- **Azure OpenAI** – LLM + embeddings  
- **Azure Cognitive Search** – PDF retrieval + vector search  
- **LangChain Tools** – brief/deep answer routing  
- **Custom logic** – grading, quizzes, relevance verification, conversation memory  

The system intelligently responds to user questions, automatically chooses the correct answer mode, and can quiz the user based on their performance.

---

## Key Features

### Dual Answering Modes (Tools)
Two LangChain tools handle user questions:

| Tool | Description |
|------|-------------|
| `brief_answer` | Short, 2–3 sentence responses for simple questions. |
| `deep_answer`  | Detailed explanations using Azure Cognitive Search + PDF citations. |

---

### Intelligent Question Grading
Each question is automatically classified as:

- **Strong**
- **Normal**
- **CounterCue**

The selected grade determines whether the tutor uses a brief or deep answer.

---

### PDF Search + Citation Engine
When `deep_answer` is triggered:

1. Embeddings are generated using Azure OpenAI.  
2. A vector search is performed on Azure Cognitive Search.  
3. Relevant PDF pages are returned.  
4. The final answer includes **strict markdown citations**.

If no matching documents are found, the system falls back to a deep LLM explanation.

---

### Automatic Quiz Generation
After three "Strong" questions, the system:

- Generates a quiz question  
- Evaluates responses  
- Provides explanations  
- Logs attempts  

---

### Persistent Memory + Logging
The system stores:

- Full conversation history  
- All events (grading, tool usage, quiz attempts)  
- Quiz states  
- Timestamps  

All persisted in **chat_log.json**.

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/tech-aakash/Agentic-AI-based-RAG.git
cd <project-folder>
```

### 2. Install dependencies
```bash
pip install --no-cache-dir -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file:

```env
AZURE_OPENAI_CHATGPT_MODEL=gpt-4o
AZURE_OPENAI_CHATGPT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_ENDPOINT=https://<your-endpoint>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>

AZURE_OPENAI_EMB_DEPLOYMENT_NAME=<embedding-name>
AZURE_SEARCH_ENDPOINT=https://<your-search>.search.windows.net
AZURE_SEARCH_INDEX_NAME=<index-name>
AZURE_SEARCH_API_KEY=<search-key>
```

### 4. Run the app
```bash
python main.py
```

---

## Project Structure

```
.
├── main.py
├── templates/
│   └── chat.html
├── chat_log.json
├── requirements.txt
└── README.md
```



## 🤝 Contributing
Pull requests are welcome!
