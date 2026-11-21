from langchain_openai import ChatOpenAI
from prompts import react_prompt
from langchain_classic.agents import create_react_agent, AgentExecutor
from tools import tools
from dotenv import load_dotenv

load_dotenv()

FALLBACK_STRING = "I couldn't work that out within the step limit. Please try rephrasing the question."

model = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)

agent = create_react_agent(model, tools, react_prompt)

followup_agent = AgentExecutor(agent = agent, tools = tools, return_intermediate_steps=True, verbose=False, max_iterations = 5, handle_parsing_errors=True)

def format_history(history):
    if not history:
        return "(none yet)"
    lines = []
    for question, answer in history:
        lines.append(f"Agent: {question}")
        lines.append(f"Assistant: {answer}")
    return "\n".join(lines)

def ask_detailed(question, product, review, history):
    chat_history = format_history(history)
    response = followup_agent.invoke({"input":question, "product":product, "review":review, "chat_history":chat_history})
    tools_used = [action.tool for action, observation in response['intermediate_steps'] if action.tool != "_Exception"]
    answer = response['output']
    if answer.startswith("Agent stopped due to"):
        answer = FALLBACK_STRING
    return {"answer": answer,"tools_used": tools_used}

def ask(question, product, review, history):
    return ask_detailed(question, product, review, history)['answer']