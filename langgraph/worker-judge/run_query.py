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
            "browser": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    )
    tools = await client.get_tools()

    worker_llm = ChatOpenAI(model=worker_model)
    judge_llm = ChatOpenAI(model=judge_model)

    worker_prompt = """
    1. Perform web searches to answer the users query.
    2. If a search tool returns an error, use a different search tool.
    3. Use the links returned by the search tools to retrieve data to answer the user.
    """

    worker = create_react_agent(
        model=worker_llm,
        tools=tools,
        prompt=worker_prompt,
    )


    judge_prompt_template_string = """
    1. You are a judge whose job it is to examine the users original query and an AI models latest response.
    2. You should judge whether to assess whether the AI model has answered the users query with information or if it is telling them to wait.
    3. User queries are generally very task specific, and all you should need to do is to remid the AI model to complete the task.
    4. If the AI model response answers the user query you should respond with the JSON object:
    {"decision": "pass"}
    5. If the AI model response does not answer the users query then you should respond with the JSON object:
    {"decision": "revise", "feedback": "Please complete the task and answer the users query"}
    6. Always return your response as a valid json object wrapped in ```json and ```

    Below is the users query, followed by the AI model response:

    <user_query>
    {{ user_query }}
    </user_query>
    <ai_model_response>
    {{ ai_model_response }}
    </ai_model_response>
    """
    judge_prompt_template = jinja2.Template(judge_prompt_template_string)


    def judge(state: OverallState):
        user_query = state["messages"][0].content
        assistant_a = state["messages"][-1].content

        judge_prompt = [
            {"role": "system", "content": "Look at the AI response and decide if it is answering the user with information or telling them to wait."},
            {"role": "user",   "content": judge_prompt_template.render(user_query=user_query, ai_model_response=assistant_a)}
        ]

        parsed = None
        for i in range(5):
            raw = judge_llm.invoke(judge_prompt).content
            try:
                parsed = json.loads(re.sub(r"```json|```", "", raw).strip())
                break
            except Exception:
                continue

        if parsed["decision"] == "pass":
            return parsed
        else:
            return parsed

    builder = StateGraph(OverallState)
    builder.add_node("worker", worker)
    builder.add_node("judge", judge)

    builder.add_edge(START, "worker")
    builder.add_edge("worker", "judge")

    def route(state: OverallState):
        return state.get("decision")

    builder.add_conditional_edges("judge", route, {
        "pass":   END,
        "revise": "worker"
    })

    graph = builder.compile()

    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": query}]},
        {"recursion_limit": 50}
    )

    print(response["messages"][-1].content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent based web search")
    parser.add_argument(
        "--worker_model",
        type=str,
        default="gpt-4o",
        required=False,
        help="The model to use for the worker."
    )
    parser.add_argument(
        "--judge_model",
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

    asyncio.run(ask_agent_team(args.query, worker_model=args.worker_model, judge_model=args.judge_model))
