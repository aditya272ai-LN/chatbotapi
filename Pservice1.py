from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

model = ChatOpenAI(model="gpt-4o-mini")
emd=OpenAIEmbeddings(model='text-embedding-3-small')

loader = DirectoryLoader("docs",glob="**/*.pdf")
list_docs=loader.load()

splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=splitter.split_documents(list_docs)

db=FAISS.from_documents(chunks,embedding=emd)
base_rtv=db.as_retriever(search_kwargs={'k':20})

bmr_retriever=BM25Retriever.from_documents(chunks)
bmr_retriever.k=20

final_retriever=EnsembleRetriever(retrievers=[base_rtv,bmr_retriever],weights=[.5,.5])

@tool
def rag_tool(query:str)->str:
    '''retrieve relevant information from pdf document
    Use this tool when user asks factual or conceptual questions'''

    docs=final_retriever.invoke(query)
    context=''
    for doc in docs:
        context+=doc.page_content
        context+='\n'
      
    return context


sp='''You are a RAG chatbot.

For factual or conceptual questions related to the PDF,
use the rag_tool to retrieve information.

Answer only using the retrieved context.
Do not use your general knowledge.

If the retrieved context does not contain enough information
to answer the question, say "Information not found".
'''
rag_agent=create_agent(model,tools=[rag_tool],system_prompt=sp)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    answer:str

#node
def chatbot(state: State):
    resp=rag_agent.invoke({'messages':state["messages"][-1].content})
    return {'answer':resp['messages'][-1].content}
    

checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile(checkpointer=checkpointer)

def chat_service(session_id: str, message: str):
    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke({"messages": HumanMessage(content=message)},config=config)
    return result['answer']
