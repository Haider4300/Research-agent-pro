import sqlite3
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool

from langchain_community.tools import (
    DuckDuckGoSearchResults,
    WikipediaQueryRun,
    ArxivQueryRun
)
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
    ArxivAPIWrapper
)

from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from langchain_ollama import ChatOllama

ddgs_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
ddgs_web_search_tool = DuckDuckGoSearchResults(
    api_wrapper=ddgs_wrapper,
    name="web_search",
    description="Search the web for current information using DuckDuckGo. Use this tool to find recent news, articles, websites, and real-time data on any topic. Returns a list of search results with titles, snippets, and URLs."
)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(
    api_wrapper=wiki_wrapper,
    name="wikipedia",
    description="Query Wikipedia for encyclopedic knowledge on people, places, events, concepts, and historical topics. Use this tool when you need detailed, factual background information or well-established knowledge. Returns article summaries from Wikipedia."
)

arxiv_wrapper = ArxivAPIWrapper()
arxiv_tool = ArxivQueryRun(
    api_wrapper=arxiv_wrapper,
    name="arxiv",
    description="Search arXiv for academic and scientific research papers. Use this tool when the query involves scientific research, technical papers, machine learning, physics, mathematics, computer science, or other academic disciplines. Returns paper titles, authors, abstracts, and publication details."
)

@tool
def get_current_datetime() -> str:
    """Return the current date and time."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y %I:%M %p")

tools = [ddgs_web_search_tool, wiki_tool, arxiv_tool, get_current_datetime]

SYSTEM_PROMPT = """You are an expert research assistant with access to web search, Wikipedia, arXiv, and a datetime tool. Your goal is to provide thorough, accurate, and well-sourced answers to user queries.

## Research Approach
- Decompose complex questions into smaller sub-queries and explore each systematically.
- Rely on **web_search** for recent events, current developments, and general information.
- Consult **wikipedia** for established facts, historical context, and foundational knowledge.
- Use **arxiv** for scientific papers, technical research, and academic publications.
- Reference **get_current_datetime** when time-sensitive context is relevant.
- Validate claims by cross-referencing across multiple sources.

## Response Standards
- Provide clear citations indicating the source of each piece of information.
- Separate well-established facts from emerging or speculative findings.
- Format information in a clear, structured, and readable manner.
- If information is insufficient or contradictory, acknowledge this transparently.
- Offer a brief summary followed by detailed findings when appropriate.
"""

def request_tool_authorization(tool_calls):
    """Request user authorization before executing tool calls."""
    print("\n" + "="*60)
    print("TOOL CALL REQUEST - Human Approval Required")
    print("="*60)

    for i, tc in enumerate(tool_calls, 1):
        print(f"\n[{i}] Tool: {tc['name']}")
        if 'args' in tc and tc['args']:
            print(f"    Arguments: {tc['args']}")

    print("\n" + "-"*60)

    while True:
        response = input("Approve? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'")


def execute_with_authorization(agent, query, config):
    """Execute agent with human-in-the-loop authorization for tool calls."""
    input_messages = [HumanMessage(content=query)]
    result = agent.invoke({"messages": input_messages}, config)

    while True:
        last_message = result["messages"][-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            authorized = request_tool_authorization(last_message.tool_calls)

            if authorized:
                print("\nApproved! Executing tools...\n")
                result = agent.invoke(result, config)
            else:
                print("\n[X] Rejected! Agent will respond without tools.\n")
                result["messages"].append(HumanMessage(
                    content="User rejected tool use. Please answer based on your knowledge without using tools."
                ))
                result = agent.invoke(result, config)
                break

        elif isinstance(last_message, AIMessage) and last_message.content:
            print(f"\n🤖 Agent Response:\n{last_message.content}\n")
            break

        elif isinstance(last_message, ToolMessage):
            result = agent.invoke(result, config)

        else:
            break


def initialize_agent():
    """Initialize the research agent."""
    llm = ChatOllama(
        model="qwen3.5:cloud",
        temperature=0
    )

    memory = MemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        system_prompt=SYSTEM_PROMPT
    )

    return agent


def display_header():
    """Display program header."""
    print("="*60)
    print("Research AI Agent with Human-in-the-Loop")
    print("="*60)
    print("Tools: Web Search, Wikipedia, arXiv, DateTime")
    print("Type 'quit' or 'exit' to end session")
    print("="*60 + "\n")


def main():
    display_header()
    agent = initialize_agent()

    config = {"configurable": {"thread_id": "research-session-1"}}

    while True:
        try:
            query = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Happy researching!")
            break

        try:
            execute_with_authorization(agent, query, config)
        except Exception as err:
            print(f"\n❌ Error: {err}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()