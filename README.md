# 🔬 Research AI Agent with Human-in-the-Loop

**Author:** Ali Haider (AI-Engineer)  
**Project:** Research Agent Pro  

---

## 📌 Overview

The Research AI Agent is a conversational AI assistant that helps you answer questions using four different tools: **DuckDuckGo Web Search**, **Wikipedia**, **arXiv academic papers**, and a **datetime tool**. The unique feature of this agent is the **human-in-the-loop** mechanism — before the agent uses any tool, it asks for your permission. You type `yes` or `no`, and only after your approval does the tool execute. This gives you complete control and transparency over what the agent does.

The agent runs **entirely on your local machine** using **Ollama** and open-source LLMs. Your data never leaves your computer. The code is built with **LangChain**, **LangGraph**, and is designed to be easy to understand and extend.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌐 Web Search | Search the internet for real-time information using DuckDuckGo |
| 📚 Wikipedia | Look up encyclopedia articles on people, places, events, and concepts |
| 📄 arXiv | Find academic research papers in physics, math, computer science, and more |
| ⏰ DateTime | Get the current date and time |
| 🧑‍⚖️ Human-in-the-Loop | Every tool call requires your manual approval (yes/no) |
| 🧠 Local LLM | Runs offline with Ollama — no API keys, no data sharing |
| 🔁 Conversation Memory | The agent remembers previous messages within a session |
| 🛑 Cancel Tool Use | You can reject any tool call; the agent will answer without it |

---

## 🏗️ How It Works

You ask a question
↓
Agent (LLM) decides which tool to use
↓
Agent requests tool call (shows you the tool name and arguments)
↓
You approve or reject
↓ (approve)
Tool executes and returns result
↓
Agent generates final answer


## 🚀 Installation & Setup (Step by Step)

**Step 1: Clone the Repository**

Open a terminal and run:
```bash
git clone git@github.com:the-schoolofai/openai-agents-sdk.git
cd Reasearch_agent_Pro

pip install uv
uv sync
ollama list
 model="qwen3.5:cloud"

uv run python research-agent-pro.py

Example it will run like 
You: What is today's date and time?

============================================================
TOOL CALL REQUEST - Human Approval Required
============================================================

[1] Tool: get_current_datetime
    Arguments: {}

------------------------------------------------------------
Approve? (yes/no): yes

Approved! Executing tools...

🤖 Agent Response:
Today is Tuesday, April 15, 2025, 02:30 PM.