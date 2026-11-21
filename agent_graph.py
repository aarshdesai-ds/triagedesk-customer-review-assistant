from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from agent import SYSTEM_RULES, NUDGE_MESSAGE, NOT_COVERED_SENTENCE, FALLBACK_STRING, MAX_ROUNDS, model_with_tools, text_of, to_messages
from tools import tools
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    nudged: bool
    rounds: int

def call_model(state: AgentState):
    messages = state['messages']
    rounds = state['rounds']
    response = model_with_tools.invoke(messages)
    return {"messages": [response], "rounds": rounds + 1}

def nudge(state: AgentState):
    return {"messages":[HumanMessage(content=NUDGE_MESSAGE)], "nudged": True}

tool_node = ToolNode(tools)

def route_after_model(state: AgentState) -> Literal['tools', 'nudge', "__end__"]:
    last = state['messages'][-1]
    if state['rounds'] >= MAX_ROUNDS:
        return END
    if last.tool_calls:
        return "tools"
    if NOT_COVERED_SENTENCE in text_of(last).lower() and not any(isinstance(m,ToolMessage) for m in state['messages']) and not state['nudged']:
        return "nudge"
    return END         

graph = StateGraph(AgentState)

graph.add_node("call_model", call_model)
graph.add_node("tools", tool_node)
graph.add_node("nudge", nudge)

graph.add_edge(START, "call_model")

graph.add_conditional_edges("call_model", route_after_model, {
    "tools": "tools",
    "nudge": "nudge", 
    END: END
})

graph.add_edge("tools", "call_model")
graph.add_edge("nudge", "call_model")

agent_graph = graph.compile()

def ask_detailed(question, product, review, history):
    system_message = SystemMessage(content=f"{SYSTEM_RULES}\nProduct:{product}\nReview:{review}")
    messages = [system_message] + to_messages(history) + [HumanMessage(content=question)]
    result = agent_graph.invoke({"messages": messages, "nudged": False, "rounds":0})
    tools_used = []
    for message in result['messages']:
        if isinstance(message,AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                tools_used.append(tool_call['name'])
    final_message = result['messages'][-1]
    if final_message.tool_calls:
        answer = FALLBACK_STRING
    else:
        answer = text_of(final_message)
    return {"answer": answer, "tools_used":tools_used}

def ask(question, product, review, history):
    return ask_detailed(question, product, review, history)['answer']