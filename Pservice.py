from langgraph.graph import StateGraph, START, END,add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated

model = ChatOpenAI(model="gpt-4o-mini")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#node
def chatbot(state: State):
    resp=model.invoke(state["messages"])
    return {'messages':[resp]}
    
checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile(checkpointer=checkpointer)

def chat_service(session_id: str, message: str):
    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke({"messages": HumanMessage(content=message)},config=config)
    return result['messages'][-1].content
