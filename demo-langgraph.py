import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, Union, List
import operator
import json
import io
from contextlib import redirect_stdout
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Ensure ANTHROPIC_API_KEY is set in your .env file
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

def create_claude_llm():
    return ChatAnthropic(model="claude-3-sonnet-20240229")

# Set up model and output parser
llm = create_claude_llm()
output_parser = StrOutputParser()

def write_code(request: str) -> str:
    logger.info(f"Writing code for request: {request}")
    code = llm.invoke(f"Write Python code for the following request: {request}")
    logger.info(f"Generated code:\n{code}")
    return code

def review_code(code: str) -> str:
    logger.info(f"Reviewing code:\n{code}")
    review = llm.invoke(f"Review the following Python code and provide feedback:\n{code}")
    logger.info(f"Review result: {review}")
    return review

def execute_code(code: str) -> str:
    logger.info(f"Executing code:\n{code}")
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            exec(code, globals())
        return f"Code executed successfully. Output:\n{output.getvalue()}"
    except Exception as e:
        error_msg = f"Error executing code: {str(e)}"
        logger.error(error_msg)
        return error_msg

def check_output(output: str) -> str:
    logger.info(f"Checking output: {output}")
    check_result = llm.invoke(f"Assess the following output and provide feedback:\n{output}")
    logger.info(f"Check result: {check_result}")
    return check_result

def task_complete(message: str) -> str:
    logger.info(f"Task completed: {message}")
    return "Task completed successfully."

# Set up agent tools
tools = [
    Tool(name="Write Code", func=write_code, description="Write Python code based on a request"),
    Tool(name="Review Code", func=review_code, description="Review Python code"),
    Tool(name="Execute Code", func=execute_code, description="Execute Python code"),
    Tool(name="Check Output", func=check_output, description="Check the output of executed code"),
    Tool(name="Task Complete", func=task_complete, description="Mark the task as complete"),
    Tool(name="LangGraph Demo", func=langgraph_demo, description="Run the LangGraph demo project")
]

tool_names = [tool.name for tool in tools]

# Set up system prompt
system_prompt = """
You are an AI assistant capable of writing, reviewing, and executing Python code.
Your task is to generate code based on a given request, review it, execute it if approved, and check its output.
Follow this process:
1) Write Code
2) Review Code
3) Execute Code
4) Check Output
Respond with the next action to take based on the current state and results.
Be explicit about each action you're taking.

Once you've completed all steps and verified the code works correctly, use the "Task Complete" action.

Always format your response as follows:
{{
  "action": "Write Code" | "Review Code" | "Execute Code" | "Check Output" | "Task Complete",
  "action_input": "Your input here"
}}

You have access to the following tools:
{tools}

Tool names: {tool_names}
"""

human_prompt = """
{input}

{agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt),
])

# Set up memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize agent
agent = create_structured_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=memory, 
    verbose=True,
    handle_parsing_errors=True
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    agent_outcome: Union[dict, str, None]
    original_request: str
    generated_code: str
    review_result: str
    execution_output: str
    chat_history: List[BaseMessage]
    current_action: str

def should_continue(state):
    if isinstance(state["agent_outcome"], dict):
        if "Task Complete" in state["agent_outcome"].get("action", ""):
            return "end"
        elif state["agent_outcome"].get("action") == "Error":
            return "end"
    return "continue"

def call_model(state):
    agent_input = {
        "input": state["original_request"],
        "agent_scratchpad": state.get("agent_scratchpad", ""),
        "chat_history": state["chat_history"]
    }
    try:
        result = agent_executor.invoke(agent_input)
        try:
            parsed_output = json.loads(result["output"])
            if "Task Complete" in parsed_output["action"]:
                return {
                    "agent_outcome": parsed_output,
                    "chat_history": memory.chat_memory.messages,
                    "current_action": "Task Complete"
                }
            else:
                return {
                    "agent_outcome": parsed_output,
                    "chat_history": memory.chat_memory.messages,
                    "current_action": parsed_output["action"]
                }
        except json.JSONDecodeError:
            logger.warning("Failed to parse agent output as JSON. Using raw output.")
            return {
                "agent_outcome": result["output"],
                "chat_history": memory.chat_memory.messages,
                "current_action": "Unknown"
            }
    except Exception as e:
        logger.error(f"Error in call_model: {str(e)}")
        return {
            "agent_outcome": {"action": "Error", "action_input": str(e)},
            "chat_history": memory.chat_memory.messages,
            "current_action": "Error"
        }

def call_tool(state):
    action = state["agent_outcome"]
    if isinstance(action, dict) and "action" in action and "action_input" in action:
        tool = next((t for t in tools if t.name == action["action"]), None)
        if tool:
            tool_result = tool.func(action["action_input"])
            return {"agent_scratchpad": f"Tool '{action['action']}' result: {tool_result}"}
    return {}

def call_set_initial_state(state):
    messages = state["messages"]
    last_message = messages[-1]
    return {
        "messages": messages,
        "agent_outcome": None,
        "original_request": last_message.content,
        "generated_code": "",
        "review_result": "",
        "execution_output": "",
        "chat_history": [],
        "current_action": "Starting"
    }

def track_progress(state):
    current_action = state.get("current_action", "Starting")
    logger.info(f"Current Action: {current_action}")
    return state

# Set up the graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("action", call_tool)
graph.add_node("initial_state", call_set_initial_state)
graph.add_node("progress", track_progress)
graph.set_entry_point("initial_state")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
graph.add_edge("action", "progress")
graph.add_edge("progress", "agent")
graph.add_edge("initial_state", "agent")

# Compile the graph
app = graph.compile()

async def main():
    feature_request = """
    Create a Python function that calculates the factorial of a number using recursion.
    """
    state = {
        "messages": [HumanMessage(content=feature_request)],
        "original_request": feature_request,
        "chat_history": []
    }
    try:
        async for chunk in app.astream(state):
            if "current_action" in chunk:
                print(f"Current Action: {chunk['current_action']}")
            if "agent_outcome" in chunk:
                if isinstance(chunk['agent_outcome'], dict):
                    print(f"Agent Output: {json.dumps(chunk['agent_outcome'], indent=2)}")
                    if "Task Complete" in chunk['agent_outcome'].get("action", ""):
                        print("Task completed. Exiting.")
                        break
                else:
                    print(f"Agent Output: {chunk['agent_outcome']}")
            if chunk.get("current_action") == "Error":
                print(f"Error occurred: {chunk['agent_outcome']['action_input']}")
                break
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        logger.error("Traceback:", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())