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
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, filter_messages
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command



class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[dict]
    working_summary: str = ""
    



async def ask_agent_team(query, basic_model="gpt-4o", thinking_model="o3-mini"):
    cfg = {"configurable": {"thread_id": "cli-thread"}, "recursion_limit": 80}
    client = MultiServerMCPClient(
        {
            "browser": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    )
    tools = await client.get_tools()

    researcher_llm = ChatOpenAI(model=thinking_model)
    scoper_llm = ChatOpenAI(model=basic_model)
    summarizer_llm = ChatOpenAI(model=basic_model)

    summarizer_prompt_template_string = """
    1. Please read through the working summary.
    2. Then read through the additional information.
    3. Re-write the working summary to incorporte the additional information.
    4. Keep things concise, but make sure that important information is saved in
    the updated summary.

    <working_summary>
    {{ working_summary }}
    </working_summary>

    <additional_information>
    {{ additional_information }}
    </additional_information>
    """
    summarizer_prompt_template = jinja2.Template(summarizer_prompt_template_string)

    researcher_prompt = """
    1. Perform web searches to answer the users query.
    2. Search iteratively, first generate search terms to look up, then use information
    gleaned from reading through the results to expand your searches to other terms.
    3. If a search tool returns an error, use a different search tool.
    4. Use the links returned by the search tools to retrieve data to answer the user.
    5. Read through the relevant results returned and pull together the information
    needed to answer the user query.
    """

    researcher_agent = create_react_agent(
        model=researcher_llm,
        tools=tools,
        prompt=researcher_prompt,
    )

    scoper_prompt = """
    1. Read the preliminary findings on the initial research done on the users query.
    2. Confirm with the user that you are researching the correct topic.
    3. Develop a sequential and iterative plan for researching the users query.
    4. Ask the user a few questions about style and level of detail, and user their
    responses to refine the plan.
    5. Return a JSON object of the form:
    {
        "needs_more_information": <true or false>,
        "question_for_user": <clarifying-questions-for-the-user>,
        "plan": [
            {
                "section": <plan-section-1-title>,
                "notes": <notes-on-how-to-execute-this-section-of-the-plan>,
                "result": None
            },
            {
                "section": <plan-section-2-title>,
                "notes": <notes-on-how-to-execute-this-section-of-the-plan>
                "result": None
            },
            ...
        ]
    }
    6. The needs_more_information key in the returned JSON object should be true if there
    is a question that the user needs to answer.
    7. The question_for_user key should contain the text of the question that the user
    needs to answer.
    8. The plan key should be a list of dictionaries each with section, notes and result keys.
    9. Initially the result for each plan should be null.
    10. The notes in each plan section should contain instructions and search terms for a researcher
    to search for using a web search tool to execute that section of the plan.
    11. Please return your JSON wrapped in ```json and ``` markers.
    """

    def scoper(state: OverallState) -> OverallState:
        researcher_messages = researcher_agent.invoke(state)
        research_result = researcher_messages["messages"][-1].content
        messages = state["messages"]
        prompt = [
            {"role": "system", "content": "Your task is to scope out a research task based on the user query."},
        ]
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                prompt.append({"role": "assistant", "content": message.content})
        
        prompt.append({"role": "assistant", "content": research_result})
        prompt.append({"role": "user", "content": scoper_prompt})
        
        for i in range(5):
            try:
                raw_response = scoper_llm.invoke(prompt).content
                parsed = json.loads(re.sub(r"```json|```", "", raw_response).strip())
                assert("needs_more_information" in parsed)
                assert("question_for_user" in parsed)
                assert("plan" in parsed)
                for section in parsed["plan"]:
                    assert("section" in section)
                    assert("notes" in section)
                    assert("result" in section)
                if parsed["needs_more_information"]:
                    return {
                        "needs_more_information": parsed["needs_more_information"],
                        "plan": parsed["plan"],
                        "messages": AIMessage(content=parsed["question_for_user"])
                    }
                else:
                    return {
                        "needs_more_information": parsed["needs_more_information"],
                        "plan": parsed["plan"]
                    }                    
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()
                continue
        
        return {
            "needs_more_information": False,
            "plan": [
                {
                    "section": "Initial Research",
                    "notes": """
                    1. Perform an initial search based on the users query.
                    2. Use the results of this search to generate a few more search terms and use them
                    to perform a few more searches.
                    3. Use the aggregate search results to answer the users query.
                    """
                }
            ]
        }

    def ask_human(state):
        clarifying_question = state["messages"][-1].content
        feedback = interrupt(f"Assistant: {clarifying_question}\nPlease provide an answer:")
        messages = state["messages"]
        messages.append(HumanMessage(content=feedback))
        return {"messages": messages}
    
    iterative_researcher_prompt_template_string = """
    1. You are to focus on researching a particular aspect of the user query.
    2. The task you are supposed to perform through your research as well as
    some notes on how you are to research it are given below.
    3. Also included is a working summary of the overall research findings.

    <working_summary>
    {{ working_summary }}
    </working_summary>

    <research_task>
    {{ research_task }}
    </research_task>
    <notes>
    {{ notes }}
    </notes>
    """
    iterative_researcher_prompt_template = jinja2.Template(iterative_researcher_prompt_template_string)

    def iterative_researcher(state):
        human_messages = filter_messages(state["messages"], include_types="human")
        researcher_messages = list(map(lambda m: {"role": "user", "content": m.content}, human_messages))
        remaining_section_indices = list(filter(lambda idx: state["plan"][idx]["result"] is None, range(len(state["plan"]))))
        current_section_idx = None
        if len(remaining_section_indices) > 0:
            current_section_idx = remaining_section_indices[0]
        
        if current_section_idx is not None:
            current_section = state["plan"][current_section_idx]
            print(f"Starting reearch on {current_section['section']}")
            research_prompt = iterative_researcher_prompt_template.render(
                working_summary=state.get("working_summary", ""),
                research_task=current_section["section"],
                notes=current_section["notes"]
            )
            researcher_messages.append({"role": "user", "content": research_prompt})
            researcher_response = researcher_agent.invoke({"messages": researcher_messages})
            research_result = researcher_response["messages"][-1].content
            state["plan"][current_section_idx]["result"] = research_result
            summarizer_prompt = summarizer_prompt_template.render(
                working_summary=state.get("working_summary", ""),
                additional_information=research_result)
            updated_working_summary = summarizer_llm.invoke([{"role": "user", "content": summarizer_prompt}]).content

            return {
                "working_summary": updated_working_summary,
                "plan": state["plan"]
            }
        
        return state
    
    def deliver_findings(state):
        print("========Research Results========")
        for section in state["plan"]:
            print(section["section"])
            print("----------------")
            print(section["result"])
            print("================")
        
        print("==============End===============")
             
    
    graph_builder = StateGraph(OverallState)
    graph_builder.add_node("scoper", scoper)
    graph_builder.add_node("ask_human", ask_human)
    graph_builder.add_node("iterative_researcher", iterative_researcher)
    graph_builder.add_node("deliver_findings", deliver_findings)

    def clarify(state):
        if state.get("needs_more_information"):
            return "ask_human"
        else:
            return "start_research"
    
    graph_builder.add_edge(START, "scoper")
    graph_builder.add_conditional_edges(
        "scoper",
        clarify,
        {
            "ask_human": "ask_human",
            "start_research": "iterative_researcher"
        }
    )
    graph_builder.add_edge("ask_human", "scoper")

    def is_research_done(state):
        unfinished_sections = list(filter(lambda s: s["result"] is None, state["plan"]))
        if len(unfinished_sections) > 0:
            return "continue_research"
        
        return "deliver_findings"
    
    graph_builder.add_conditional_edges(
        "iterative_researcher",
        is_research_done,
        {
            "continue_research": "iterative_researcher",
            "deliver_findings": "deliver_findings"
        }
    )
    graph_builder.add_edge("deliver_findings", END)

    graph = graph_builder.compile(checkpointer=InMemorySaver())

    response = await graph.ainvoke(
        {"messages": [{"role": "user", "content": query}]},
        cfg 
    )

    while "__interrupt__" in response:
        interrupts = response["__interrupt__"]
        if not isinstance(interrupts, list) or len(interrupts) == 0:
            raise RuntimeError("Unexpected interrupt payload")

        if len(interrupts) == 1:
            prompt = getattr(interrupts[0], "value", None) or "Input:"
            # block in a thread so we don't stall the event loop
            user_reply = await asyncio.to_thread(input, f"{prompt} ")
            response = await graph.ainvoke(Command(resume=user_reply), cfg)
        else:
            reply_map = {}
            for intr in interrupts:
                prompt = getattr(intr, "value", None) or "Input:"
                ans = await asyncio.to_thread(input, f"{prompt} ")
                intr_id = getattr(intr, "id", None) or getattr(intr, "interrupt_id", None) or intr["id"]
                reply_map[intr_id] = ans
            response = await graph.ainvoke(Command(resume=reply_map), cfg)

        



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
        default="o3-mini",
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

    asyncio.run(ask_agent_team(args.query, basic_model=args.basic_model, thinking_model=args.thinking_model))
