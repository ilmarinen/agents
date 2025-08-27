import argparse
import json
import re
import jinja2
import asyncio
from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    decision: str



async def ask_agent_team(query, worker_model="gpt-4o", judge_model="gpt-4o"):
    client = MultiServerMCPClient(
        {
            "browser_tools": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    )
    tools = await client.get_tools()

    worker_llm = ChatOpenAI(model=worker_model)
    judge_llm = ChatOpenAI(model=judge_model)

    worker_prompt = """
    1. Follow the users instructions.
    2. Use the tools available to follow the users instructions.
    """

    worker = create_react_agent(
        model=worker_llm,
        tools=tools,
        prompt=worker_prompt,
    )

    response = await worker.ainvoke(
        {"messages": [{"role": "user", "content": query}]},
        {"recursion_limit": 50}
    )

    print(response["messages"][-1].content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent based web search")
    parser.add_argument(
        "--basic_model",
        type=str,
        default="gpt-4o",
        required=False,
        help="The model to use for the worker."
    )
    parser.add_argument(
        "--thinking_model",
        type=str,
        default="gpt-4o",
        required=False,
        help="The model to use for the judge."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        required=True,
        help="The query that you would like the answer to."
    )
    args = parser.parse_args()

    asyncio.run(ask_agent_team(args.query, worker_model=args.basic_model, judge_model=args.thinking_model))
