import re
import json
import argparse
import jinja2
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import mcp_server_tools, SseServerParams
from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, UserMessage, LLMMessage, ModelFamily
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_agentchat.teams import (
    DiGraphBuilder,
    GraphFlow,
)


async def main(user_query, worker_model, judge_model) -> None:
    # Create server params for the remote MCP service
    server_params = SseServerParams(
        url="http://localhost:8000/sse",
        # headers={"Authorization": "Bearer your-api-key", "Content-Type": "application/json"},
        timeout=30,  # Connection timeout in seconds
    )

    # Get the translation tool from the server
    tools = await mcp_server_tools(server_params)

    # Create an agent that can use the translation tool
    worker_model_client = OpenAIChatCompletionClient(model=worker_model)
    judge_model_client = OpenAIChatCompletionClient(model=judge_model)

    worker_prompt = """
    1. Perform web searches to answer the users query.
    2. If a search tool returns an error, use a different search tool.
    3. Use the links returned by the search tools to retrieve data to answer the user.
    """
    worker = AssistantAgent(
        name="worker",
        model_client=worker_model_client,
        tools=tools,
        system_message=worker_prompt,
    )

    judge_prompt = """
    1. You are a judge whose job it is to examine the users original query and an AI models latest response.
    2. You should judge whether to assess whether the AI model has answered the users query with information or if it is telling them to wait.
    3. User queries are generally very task specific, and all you should need to do is to remid the AI model to complete the task.
    4. If the AI model response answers the user query you should respond with the JSON object:
    {"decision": "pass"}
    5. If the AI model response does not answer the users query then you should respond with the JSON object:
    {"decision": "revise", "feedback": "Please complete the task and answer the users query"}
    6. Always return your response as a valid json object wrapped in ```json and ```
    """
    judge_core_agent = AssistantAgent(
        name="judge",
        model_client=judge_model_client,
        system_message=judge_prompt,
    )
    judge = MessageFilterAgent(
        name="judge",
        wrapped_agent=judge_core_agent,
        filter=MessageFilterConfig(
            per_source=[
                PerSourceFilter(source="user", position="first", count=1),
                PerSourceFilter(source="worker", position="last", count=1),
            ]
        )
    )

    answer_core_agent = AssistantAgent(
        name="answer_agent",
        model_client=judge_model_client,
        system_message="Answer the users query using the answer provided by the worker.",
    )
    answer_agent = MessageFilterAgent(
        name="answer_agent",
        wrapped_agent=answer_core_agent,
        filter=MessageFilterConfig(
            per_source=[
                PerSourceFilter(source="user", position="first", count=1),
                PerSourceFilter(source="worker", position="last", count=1),
            ]
        )
    )

    def revise_condition(message):
        raw = message.content
        parsed = json.loads(re.sub(r"```json|```", "", raw).strip())
        return (parsed["decision"] == "revise")

    def answer_condition(message):
        raw = message.content
        parsed = json.loads(re.sub(r"```json|```", "", raw).strip())
        return (parsed["decision"] == "pass")

    builder = DiGraphBuilder()
    builder.add_node(worker).add_node(judge).add_node(answer_agent)
    builder.add_edge(worker, judge)
    builder.add_edge(judge, worker, revise_condition)
    builder.add_edge(judge, answer_agent, answer_condition)
    builder.set_entry_point(worker)

    max_msg_termination = MaxMessageTermination(max_messages=10)

    flow = GraphFlow(
        participants=[worker, judge, answer_agent],
        graph=builder.build(),
        termination_condition=max_msg_termination,
    )

    # Let the agent translate some text
    await Console(
        flow.run_stream(task=user_query, cancellation_token=CancellationToken())
    )


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

    asyncio.run(main(args.query, args.worker_model, args.judge_model))
