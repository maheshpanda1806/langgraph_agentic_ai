"""
Human-in-the-Loop (HITL) Stock Trading Application
This module demonstrates a LangGraph-based application that incorporates human approval
for stock trading decisions using the interrupt/resume pattern.
"""

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import required libraries for LangChain and LangGraph
from langchain.chat_models import init_chat_model
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# Define the state structure for the conversation
class State(TypedDict):
    """State TypedDict to manage messages in the graph"""
    messages: Annotated[list, add_messages]

# Tool function to get stock prices
@tool
def get_stock_price(symbol: str) -> float:
    '''Return the current price of a stock given the stock symbol'''
    # Mock stock prices for demonstration
    return {"MSFT": 200.3, "AAPL": 100.4, "AMZN": 150.0, "RIL": 87.6}.get(symbol, 0.0)

# Tool function to buy stocks with human approval
@tool
def buy_stocks(symbol: str, quantity: int, total_price: float) -> str:
    '''Buy stocks given the stock symbol and quantity'''
    # Interrupt the execution and wait for human approval
    decision = interrupt(f"Approve buying {quantity} {symbol} stocks for ${total_price:.2f}?")

    # Process the human decision
    if decision == "yes":
        return f"You bought {quantity} shares of {symbol} for a total price of {total_price}"
    else:
        return "Buying declined."


# Initialize tools list
tools = [get_stock_price, buy_stocks]

# Initialize the LLM and bind tools to it
llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)

# Define the chatbot node function that invokes the LLM
def chatbot_node(state: State):
    """Process state through the LLM and return updated state"""
    msg = llm_with_tools.invoke(state["messages"])
    return {"messages": [msg]}

# Initialize memory saver for persistence
memory = MemorySaver()

# Build the state graph
builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))

# Define graph edges
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)  # Conditionally route to tools
builder.add_edge("tools", "chatbot")  # After tools, return to chatbot
builder.add_edge("chatbot", END)

# Compile the graph with memory checkpointing
graph = builder.compile(checkpointer=memory)

# Configuration for persistent thread
config = {"configurable": {"thread_id": "buy_thread"}}

# Step 1: User asks for stock price
print("=== Step 1: Get Stock Price ===")
state = graph.invoke({"messages":[{"role":"user","content":"What is the current price of 10 MSFT stocks?"}]}, config=config)
print(state["messages"][-1].content)

# Step 2: User asks to buy stocks (triggers interruption)
print("\n=== Step 2: Request to Buy (Awaiting Approval) ===")
state = graph.invoke({"messages":[{"role":"user","content":"Buy 10 MSFT stocks at current price."}]}, config=config)
print(state.get("__interrupt__"))

# Step 3: Human provides approval decision
print("\n=== Step 3: Resume with Human Decision ===")
decision = input("Approve (yes/no): ")
state = graph.invoke(Command(resume=decision), config=config)
print(state["messages"][-1].content)



