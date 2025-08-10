import argparse
import json
import random
import uuid
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
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from neo4j import GraphDatabase


URI = "neo4j://localhost:7687/"
AUTH = ("neo4j", "password")


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    root_node_name: str
    route_to: str


def record_to_dict(record):
    if "Hypothesis" in record["h"].labels:
        return {
            "label": "Hypothesis",
            "name": record["h"].get("name"),
            "text": record["h"].get("text"),
            "active": record["h"].get("active"),
            "explored": record["h"].get("explored"),
            "exploration": record["h"].get("exploration"),
            "solution": record["h"].get("solution"),
            "solved": record["h"].get("solved"),
            "verified": record["h"].get("verified"),
            "dead_end": record["h"].get("dead_end")
        }

    return {
        "label": "Root",
        "name": record["h"].get("name"),
        "active": record["h"].get("active"),
    }

def delete_all_nodes_and_edges():
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        summary = driver.execute_query("""
            MATCH (a)-[r]->(b)
            DELETE r
            """,
        ).summary
        summary = driver.execute_query("""
            MATCH (a)
            DELETE a
            """,
        ).summary


def create_root_node(root_node_name):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        summary = driver.execute_query("""
            MERGE (n:Root {name: $root_node_name, active: true})
            RETURN n
            """,
            root_node_name=root_node_name
        ).summary


def get_root_node(root_node_name):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, summary, keys = driver.execute_query("""
            MATCH (h:Root {name: $root_node_name})
            RETURN h
            """,
            root_node_name=root_node_name
        )
        records = list(records)
        if len(records) > 0:
            return record_to_dict(records[0])
        
        return None


def mark_root_inactive():
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, summary, keys = driver.execute_query("""
            MATCH (h:Root)
            SET h.active = false
            RETURN h
            """,
            hypothesis_name=hypothesis_name,
            exploration_text=exploration_text,
        )
    return list(map(lambda r: record_to_dict(r), records)).pop()


def add_hypothesis(parent_node, hypothesis_text):
    parent_node_name = parent_node["name"]
    name = str(uuid.uuid4())
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        if parent_node["label"] == "Root":
            records, summary, keys = driver.execute_query("""
                MATCH (p:Root {name: $parent_node_name})
                CREATE (h:Hypothesis {name: $name, text: $text, explored: false, exploration: null, active: false, solution: null, solved: false, dead_end: false, verified: false})
                CREATE (p)-[:NEXT]->(h)
                RETURN h
                """,
                parent_node_name=parent_node_name,
                name=name, text=hypothesis_text,
            )
        else:
            records, summary, keys = driver.execute_query("""
                MATCH (p:Hypothesis {name: $parent_node_name})
                CREATE (h:Hypothesis {name: $name, text: $text, explored: false, exploration: null, active: false, solution: null, solved: false, dead_end: false, verified: false})
                CREATE (p)-[:NEXT]->(h)
                RETURN h
                """,
                parent_node_name=parent_node_name, name=name, text=hypothesis_text,
            )


def mark_hypothesis_explored(hypothesis_node, exploration_text):
    hypothesis_name = hypothesis_node["name"]
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, summary, keys = driver.execute_query("""
            MATCH (h:Hypothesis {name: $hypothesis_name})
            SET h.explored = true
            SET h.exploration = $exploration_text
            RETURN h
            """,
            hypothesis_name=hypothesis_name,
            exploration_text=exploration_text,
        )
    return list(map(lambda r: record_to_dict(r), records)).pop()


def add_solution_text(hypothesis_node, solution_text):
    hypothesis_name = hypothesis_node["name"]
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, summary, keys = driver.execute_query("""
            MATCH (h:Hypothesis {name: $hypothesis_name})
            SET h.solution = $solution_text
            RETURN h
            """,
            hypothesis_name=hypothesis_name,
            solution_text=solution_text,
        )
    return list(map(lambda r: record_to_dict(r), records)).pop()


def mark_hypothesis_dead_end(node):
    hypothesis_name = node.get("name")
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, summary, keys = driver.execute_query("""
            MATCH (h:Hypothesis {name: $hypothesis_name})
            SET h.dead_end = true
            RETURN h
            """,
            hypothesis_name=hypothesis_name,
        )
    return list(map(lambda r: record_to_dict(r), records)).pop()


def mark_hypothesis_verified(hypothesis_node, solved):
    hypothesis_name = hypothesis_node["name"]
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        if solved:
            records, summary, keys = driver.execute_query("""
                MATCH (h:Hypothesis {name: $hypothesis_name})
                SET h.solved = true
                SET h.verified = true
                RETURN h
                """,
                hypothesis_name=hypothesis_name,
            )
            return list(map(lambda r: record_to_dict(r), records)).pop()
        else:
            records, summary, keys = driver.execute_query("""
                MATCH (h:Hypothesis {name: $hypothesis_name})
                SET h.solved = false
                SET h.verified = true
                RETURN h
                """,
                hypothesis_name=hypothesis_name,
            )
            return list(map(lambda r: record_to_dict(r), records)).pop()


def get_active_node(root_node_name):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, summary, keys = driver.execute_query("""
            MATCH (root:Root {name: $root_node_name})-[:NEXT *]->(h {active: true})
            RETURN h
            """,
            root_node_name=root_node_name
        )
    matching_nodes = list(map(lambda r: record_to_dict(r), records))
    if len(matching_nodes) > 0:
        return matching_nodes[0]
    else:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            records, summary, keys = driver.execute_query("""
                MATCH (h:Root {name: $root_node_name, active: true})
                RETURN h
                """,
                root_node_name=root_node_name
            )
        matching_nodes = list(map(lambda r: record_to_dict(r), records))
        if len(matching_nodes) > 0:
            return matching_nodes[0]

    return None


def get_all_leaf_nodes():
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, summary, keys = driver.execute_query("""
            MATCH (root:Root)-[:NEXT *]->(leaf:Hypothesis)
            WITH leaf
            WHERE NOT (leaf)-[:NEXT]->()
            RETURN leaf
            """,
        )
    matching_hypotheses = list(map(lambda r: record_to_dict(r), records))
    
    return matching_hypotheses


def all_leaf_nodes_are_dead_ends():
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, summary, keys = driver.execute_query("""
            MATCH (root:Root)-[:NEXT *]->(leaf:Hypothesis)
            WITH leaf
            WHERE NOT (leaf)-[:NEXT]->()
            RETURN leaf
            """
        )
    matching_hypotheses = list(map(lambda r: record_to_dict(r), records))
    non_dead_ends = list(filter(lambda n: not n["dead_end"], matching_hypotheses))
    
    return (len(non_dead_ends) == 0)


def get_tree_depth():
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, summary, keys = driver.execute_query("""
            MATCH p=(parent:Root)-[:NEXT *]->(child)
            WHERE NOT (child)-[:NEXT]->()
            RETURN LENGTH(p) AS maxlength
            ORDER BY maxlength DESC
            LIMIT 1
            """,
        )
    
    return records[0]["maxlength"]


def get_node_depth(root_node_name, node):
    if node.get("label") == "Hypothesis":
        name = node.get("name")
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            records, summary, keys = driver.execute_query("""
                MATCH p=(parent:Root {name: $root_node_name})-[:NEXT *]->(child {name: $name})
                RETURN LENGTH(p) AS nodedepth
                """,
                root_node_name=root_node_name,
                name=name
            )
        
        return records[0]["nodedepth"]
    else:
        return 0


def get_children(node):
    name = node.get("name")
    if node["label"] == "Root":
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            records, summary, keys = driver.execute_query("""
                MATCH (root:Root {name: $name})-[:NEXT]->(h:Hypothesis)
                return h
                """,
                name=name
            )
            return list(map(lambda r: record_to_dict(r), records))
    else:
        name = node["name"]
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            records, summary, keys = driver.execute_query("""
                MATCH (parent:Hypothesis {name: $name})-[:NEXT]->(h:Hypothesis)
                return h
                """,
                name=name
            )
            return list(map(lambda r: record_to_dict(r), records))


def mark_node_inactive(node):
    name = node.get("name")
    if node["label"] == "Root":
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            records, summary, keys = driver.execute_query("""
                MATCH (root:Root {name: $name})
                SET root.active = false
                return root
                """,
                name=name
            )
    else:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            records, summary, keys = driver.execute_query("""
                MATCH (n:Hypothesis {name: $name})
                SET n.active = false
                return n
                """,
                name=name
            )


def mark_node_active(node):
    name = node.get("name")
    if node["label"] == "Root":
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            records, summary, keys = driver.execute_query("""
                MATCH (root:Root {name: $name})
                SET root.active = true
                return root
                """,
                name=name
            )
    else:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            records, summary, keys = driver.execute_query("""
                MATCH (n:Hypothesis {name: $name})
                SET n.active = true
                return n
                """,
                name=name
            )


def get_next_unexplored_with_common_ancestor(active_node):
    name = active_node["name"]
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, summary, keys = driver.execute_query("""
            MATCH q=(a:Hypothesis {name: $name})<-[*]-(ancestor)-[*]->(h:Hypothesis {explored: false})
            return h
            ORDER BY LENGTH(q) ASC
            LIMIT 1
            """,
            name=name
        )
        unexplored_nodes = list(map(lambda r: record_to_dict(r), records))
        if len(unexplored_nodes) > 0:
            return unexplored_nodes.pop()
    
    return None


def get_next_non_dead_end_node_with_common_ancestor(active_node):
    name = active_node["name"]
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, summary, keys = driver.execute_query("""
            MATCH q=(a:Hypothesis {name: $name})<-[*]-(ancestor)-[*]->(h:Hypothesis {dead_end: false, solution: null})
            return h
            ORDER BY LENGTH(q) ASC
            LIMIT 1
            """,
            name=name
        )
        non_dead_end_nodes = list(map(lambda r: record_to_dict(r), records))
        if len(non_dead_end_nodes) > 0:
            return non_dead_end_nodes.pop()
    
    return None


def get_path_from_root(root_node_name, active_node):
    name = active_node["name"]
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, summary, keys = driver.execute_query("""
            MATCH p=(root:Root {name: $root_node_name})-[:NEXT *]->(b {name: $name})
            return p
            LIMIT 1
            """,
            root_node_name=root_node_name,
            name=name
        )
        path = records[0]
        nodes = path["p"].nodes[1:]
        node_results = []
        for node in nodes:
            node_results.append({
                "label": node.get("label"),
                "name": node.get("name"),
                "text": node.get("text"),
                "active": node.get("active"),
                "explored": node.get("explored"),
                "exploration": node.get("exploration"),
                "solution": node.get("solution"),
                "solved": node.get("solved"),
                "verified": node.get("verified"),
                "dead_end": node.get("dead_end")
            })
        return node_results


def get_solution_node(root_node_name):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        records, summary, keys = driver.execute_query("""
            MATCH (r:Root {name: $root_node_name})-[:NEXT *]->(h:Hypothesis {solved: true})
            return h
            """,
            root_node_name=root_node_name
        )
        if len(records) == 0:
            return None

        return list(map(lambda r: record_to_dict(r), records)).pop()


async def ask_agent_team(query, basic_model="gpt-4o", reasoner_model="o3-mini"):
    cfg = {"configurable": {"thread_id": "cli-thread"}, "recursion_limit": 80}

    scoper_llm = ChatOpenAI(model=basic_model)
    hypothesizer_llm = ChatOpenAI(model=reasoner_model)
    explorer_llm = ChatOpenAI(model=reasoner_model)
    verifier_llm = ChatOpenAI(model=reasoner_model)
    narrator_llm = ChatOpenAI(model=basic_model)

    scoper_prompt = """
    1. Decide whether the user query clearly enough defined for work to begin.
    2. If more information is needed from the user return True together with a clarifying question.
    3. If no more information is needed from the user then return False.
    4. If you think that more information is needed then return your response as a valid JSON object of the form:
       {"more_info_needed": true, "clarifying_question": "<your-clarifying-question>"}
    5. If you think that no additional information is needed then return your response as a valid JSON object of the form:
       {"more_info_needed": false}
    6. Always return your response as a valid json object wrapped in ```json and ```
    """

    hypothesizer_prompt = """
    1. Look at the user query and the follow up user responses to any clarifying questions.
    2. Suggest three or fewer distinct hypotheses that should be explored in order to answer the user query.
    3. Describe each of your hypotheses in free text and return within valid JSON list of strings as follows:
    {"hypotheses": ["<hypothesis-1-text-description>", "<hypothesis-2-text-description>", "<hypothesis-3-text-description>"]}
    4. Always return your response as a valid json object wrapped in ```json and ```
    """

    explorer_prompt = """
    1. Your task is to take a close look at the user query from the perspective of the current working hypothesis.
    2. You should then attempt to verify whether the hypothesis is true or falsify it.
    3. Try to develop and sharpen or enhance the hypothesis as much as you can.
    """

    verifier_prompt = """
    1. Your task is to take a close look at the user quewry and a proposed solution to it.
    2. You should read the solution with a critical eye and determine whether or not the solution is correct.
    """

    narrator_prompt = """
    1. Your task is to go over the line of exploration that led to the solution.
    2. You should also review the solution to the users query in detail.
    3. You should make sure to use the solution to fully answer the users query.
    """

    def scoper(state):
        messages = state["messages"]

        prompt = [
            {"role": "system", "content": scoper_prompt},
        ]
        for message in messages:
            prompt.append({"role": "user", "content": message.content})

        parsed = None
        for i in range(5):
            raw = scoper_llm.invoke(prompt).content
            try:
                parsed = json.loads(re.sub(r"```json|```", "", raw).strip())
                assert("more_info_needed" in parsed)
                if parsed["more_info_needed"]:
                    ai_message = AIMessage(content=parsed["clarifying_question"])
                    messages.append(ai_message)

                return {
                    "more_info_needed": parsed["more_info_needed"],
                    "messages": messages
                }
            except Exception:
                continue

        return {"more_info_needed": True}

    def hypothesizer(state):
        root_node_name = state.get("root_node_name")
        active_node = get_active_node(root_node_name)
        if get_node_depth(root_node_name, active_node) >= 3:
            mark_hypothesis_dead_end(active_node)
            return dict()

        user_messages = state["messages"]
        prompt = [
            {"role": "system", "content": hypothesizer_prompt},
        ]
        for message in user_messages:
            prompt.append({"role": "user", "content": message.content})
        if active_node["label"] == "Hypothesis":
            exploration = active_node["exploration"]
            prompt.append({
                "role": "assistant",
                "content": exploration
            })

        prompt.append({
            "role": "user",
            "content": """
            Can you suggest three additional new and distinct hypotheses to take things forward from the current exploration.
            Please return as a JSON object formatted as follows:
            {"hypotheses": ["<hypothesis-1-text-description>", "<hypothesis-2-text-description>", "<hypothesis-3-text-description>"]}
            """
        })


        parsed = None
        for i in range(5):
            raw = hypothesizer_llm.invoke(prompt).content
            try:
                parsed = json.loads(re.sub(r"```json|```", "", raw).strip())
                assert("hypotheses" in parsed)
                assert(len(parsed["hypotheses"]) == 3)
                for hypothesis in parsed["hypotheses"]:
                    add_hypothesis(active_node, hypothesis)
                return dict()
            except Exception as e:
                print(e)
                continue
        
        if active_node["label"] == "Hypothesis":
            mark_hypothesis_dead_end(active_node)

        return dict()

    def explorer(state):
        root_node_name = state.get("root_node_name")
        active_node = get_active_node(root_node_name)
        assert(active_node["label"] == "Hypothesis")
        hypothesis_text = active_node["text"]
        user_messages = state["messages"]
        prompt = [
            {"role": "system", "content": explorer_prompt},
        ]
        for message in user_messages:
            prompt.append({"role": "user", "content": message.content})

        prompt.append({
            "role": "assistant",
            "content": f"""
            The current working hypothesis is:
            {hypothesis_text}
            """
        })

        prompt.append({
            "role": "user",
            "content": "Please analyze the original user query with respect to the current working hypothesis. Is it valid? Is it invalid? Is this direction a dead end?"
        })
        exploration_text = explorer_llm.invoke(prompt).content
        mark_hypothesis_explored(active_node, hypothesis_text)
        prompt.append({
            "role": "assistant",
            "content": exploration_text
        })
        prompt.append({
            "role": "user",
            "content": """
            Is the current working hypothesis and your exploration around it developed enough for you to attempt a solution? Or do you need to explore some more? Or is this a dead end?
            Please respond with a single JSON object {"decision": <"solution" | "explore_more" | "dead_end">}
            """
        })

        parsed = None
        for i in range(5):
            raw = explorer_llm.invoke(prompt).content
            try:
                parsed = json.loads(re.sub(r"```json|```", "", raw).strip())
                assert("decision" in parsed)
                assert(parsed["decision"] in ["solution", "explore_more", "dead_end"])
            except Exception:
                continue
        
        if parsed is None or parsed["decision"] == "dead_end":
            mark_hypothesis_dead_end(active_node)
            return dict()
        
        if parsed["decision"] == "solution":
            prompt.append({
                "role": "user",
                "content": """
                Please work out a solution to the users original querty, using the discussion thus far.
                """
            })
            solution_text = explorer_llm.invoke(prompt).content
            add_solution_text(active_node, solution_text)

        return dict()

    def verifier(state):
        root_node_name = state.get("root_node_name")
        active_node = get_active_node(root_node_name)
        assert(active_node["label"] == "Hypothesis")
        hypothesis_text = active_node["text"]
        exploration_text = active_node["exploration"]
        solution_text = active_node["solution"]

        user_messages = state["messages"]
        prompt = [
            {"role": "system", "content": verifier_prompt},
        ]
        for message in user_messages:
            prompt.append({"role": "user", "content": message.content})

        for message in user_messages:
            prompt.append({"role": "user", "content": message.content})

        prompt.append({
            "role": "assistant",
            "content": f"""
            The current working hypothesis is:
            {hypothesis_text}
            """
        })
        prompt.append({
            "role": "assistant",
            "content": exploration_text
        })
        prompt.append({
            "role": "assistant",
            "content": f"""
            And here is an attempted solution:
            {solution_text}
            """
        })
        prompt.append({
            "role": "user",
            "content": """
            Please analyze the solution with respect to the original user query and determine whether or not it solves the original user query.
            Please return a valid JSON object of the form {"solved": true} if it is a valid solution, and {"solved": false} if it is not a valid solution.
            """
        })

        parsed = None
        for i in range(5):
            raw = verifier_llm.invoke(prompt).content
            try:
                parsed = json.loads(re.sub(r"```json|```", "", raw).strip())
                assert("solved" in parsed)
                break
            except Exception:
                continue
        
        if parsed is None or not parsed["solved"]:
            mark_hypothesis_verified(active_node, False)
        else:
            mark_hypothesis_verified(active_node, True)


        return dict()

    def ask_human(state):
        # print("---ask_human---")
        clarifying_question = state["messages"][-1].content
        print(f"Assistant: {clarifying_question}")
        feedback = interrupt("Please provide an answer:")
        messages = state["messages"]
        messages.append(HumanMessage(content=feedback))
        return {"messages": messages}
    
    def delegator(state):
        if state.get("root_node_name") is None:
            root_node_name = str(uuid.uuid4())
            create_root_node(root_node_name)
            print(root_node_name)
            return {
                "root_node_name": root_node_name,
                "route_to": "hypothesizer"
            }
        
        root_node_name = state.get("root_node_name")
        root_node = get_root_node(root_node_name)

        active_node = get_active_node(root_node_name)
        children = get_children(active_node)
        if len(children) > 0:
            mark_node_inactive(active_node)
            next_active_node = random.choice(children)
            mark_node_active(next_active_node)
            return {
                "route_to": "explorer"
            }
        elif len(children) == 0 and not active_node["explored"]:
            return {
                "route_to": "explorer"
            }
        elif len(children) == 0 and active_node["explored"] and active_node["solution"] is None and not active_node["dead_end"]:
            return {
                "route_to": "hypothesizer"
            }
        elif len(children) == 0 and active_node["explored"] and active_node["solution"] is None and active_node["dead_end"]:
            next_active_node = None
            next_unexplored_node = get_next_unexplored_with_common_ancestor(active_node)
            next_non_dead_end_node = get_next_non_dead_end_node_with_common_ancestor(active_node)
            if next_unexplored_node is not None:
                next_active_node = next_unexplored_node
            elif next_non_dead_end_node is not None:
                next_active_node = next_non_dead_end_node
            mark_node_inactive(active_node)
            if next_active_node is not None:
                mark_node_active(next_active_node)
                return {
                    "route_to": "explorer"
                }
            return {
                "route_to": "narrator"
            }
        elif len(children) == 0 and active_node["explored"] and active_node["solution"] is not None and not active_node["verified"]:
            return {
                "route_to": "verifier"
            }
        elif len(children) == 0 and active_node["explored"] and active_node["solution"] is not None and active_node["verified"] and active_node["solved"]:
            return {
                "route_to": "narrator"
            }
        elif len(children) == 0 and active_node["explored"] and active_node["solution"] is not None and active_node["verified"] and not active_node["solved"]:
            next_active_node = None
            next_unexplored_node = get_next_unexplored_with_common_ancestor(active_node)
            next_non_dead_end_node = get_next_non_dead_end_node_with_common_ancestor(active_node)
            if next_unexplored_node is not None:
                next_active_node = next_unexplored_node
            elif next_non_dead_end_node is not None:
                next_active_node = next_non_dead_end_node
            mark_node_inactive(active_node)
            if next_active_node is not None:
                mark_node_active(next_active_node)
                return {
                    "route_to": "explorer"
                }
            return {
                "route_to": "narrator"
            }

    def narrator(state):
        messages = state["messages"]
        solution_node = get_solution_node(state["root_node_name"])
        if solution_node is not None:
            narrative_path = get_path_from_root(state["root_node_name"], solution_node)
            explorations = [node["exploration"] for node in narrative_path]
            explorations_consolidated = "\n\n".join(explorations)
            solution = solution_node["solution"]
            prompt = [
                {"role": "system", "content": narrator_prompt},
            ]
            for message in messages:
                if isinstance(message, HumanMessage):
                    prompt.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    prompt.append({"role": "assistant", "content": message.content})
            prompt.append({
                "role": "assistant",
                "content": f"""
                The explorations so far are as follows:
                {explorations_consolidated}

                The solution is:
                {solution}
                """
            })
            prompt.append({
                "role": "user",
                "content": "Please summarize the exploration and provide the full solution with an explanation to the user."
            })
            narrator_response = narrator_llm.invoke(prompt).content
            messages.append(AIMessage(content=narrator_response))
        else:
            messages.append(AIMessage(content="I wasn't able to solve the problem."))
        
        return dict()
    
    def clarify(state):
        if state["more_info_needed"]:
            return "more_information_needed"
        return "no_more_information_needed"

    builder = StateGraph(OverallState)
    builder.add_node("scoper", scoper)
    builder.add_node("ask_human", ask_human)
    builder.add_node("delegator", delegator)
    builder.add_node("hypothesizer", hypothesizer)
    builder.add_node("explorer", explorer)
    builder.add_node("verifier", verifier)
    builder.add_node("narrator", narrator)

    builder.add_edge(START, "scoper")
    builder.add_conditional_edges("scoper", clarify, {
        "more_information_needed": "ask_human",
        "no_more_information_needed": "delegator"
    })
    builder.add_edge("ask_human", "scoper")

    def router(state: OverallState):
        return state.get("route_to")

    builder.add_conditional_edges("delegator", router, {
        "hypothesizer": "hypothesizer",
        "explorer": "explorer",
        "verifier": "verifier",
        "narrator": "narrator"
    })

    builder.add_edge("hypothesizer", "delegator")
    builder.add_edge("explorer", "delegator")
    builder.add_edge("verifier", "delegator")

    graph = builder.compile(checkpointer=InMemorySaver())

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
        "--reasoner_model",
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

    asyncio.run(ask_agent_team(args.query, basic_model=args.basic_model, reasoner_model=args.reasoner_model))
