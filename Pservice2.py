from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# ============================================================
# 1. MODEL
# ============================================================

model = ChatOpenAI(model="gpt-4o-mini")

# ============================================================
# 2. RAG - LOAD PDF DOCUMENTS
# ============================================================

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

loader = DirectoryLoader("docs",glob="**/*.pdf",loader_cls=PyPDFLoader)
documents = loader.load()

# ============================================================
# 3. TEXT SPLITTING
# ============================================================

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = splitter.split_documents(documents)

# ============================================================
# 4. FAISS RETRIEVER
# ============================================================

vector_db = FAISS.from_documents(chunks,embedding=embeddings)
faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 20})

# ============================================================
# 5. BM25 RETRIEVER
# ============================================================

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 20

# ============================================================
# 6. ENSEMBLE RETRIEVER
# ============================================================

final_retriever = EnsembleRetriever(retrievers=[faiss_retriever,bm25_retriever],weights=[0.5,0.5])

# ============================================================
# 7. RAG TOOL
# ============================================================

@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from PDF documents.

    Use this tool when the user asks about product policies,
    product documentation, warranty, refund, return policy,
    shipping policy, product features or other information
    contained in the PDF documents.
    """

    docs = final_retriever.invoke(query)

    if not docs:
        return "Information not found"

    context = ""

    for doc in docs:

        context += doc.page_content
        context += "\n\n"

    return context


# ============================================================
# 8. RAG AGENT
# ============================================================

rag_system_prompt = """
You are a customer-support RAG assistant.

Your knowledge source is the PDF documents available through
the rag_tool.

RULES:

1. For questions about policies, product documentation,
   warranty, refund, returns, shipping, or product information,
   use rag_tool.

2. Answer ONLY using information retrieved from rag_tool.

3. Do NOT use your general knowledge.

4. Do NOT invent information.

5. If the retrieved context does not contain enough information
   to answer the question, respond exactly:

Information not found

6. Give a clear and concise answer to the customer.
"""

rag_agent = create_agent(model=model,tools=[rag_tool],system_prompt=rag_system_prompt)

# ============================================================
# 9. SQL DATABASE
# ============================================================

sql_db = SQLDatabase.from_uri("sqlite:///db/voltedge.db")

# ============================================================
# 10. SQL AGENT
# ============================================================

sql_agent = create_sql_agent(
    llm=model,
    db=sql_db,
    agent_type="tool-calling",
    prefix="""
You are a customer-support SQL database assistant.

The database contains product-related customer data,
including orders and returns.

Use the SQL database to answer questions about actual
stored records.

The database may contain information such as:

- Orders
- Returns
- Products
- Customers
- Order status
- Return status
- Order dates
- Return dates
- Product names
- Quantities
- Prices
- Amounts

IMPORTANT RULES:

1. Always query the database for questions about actual
   orders, returns, products, customers or transaction data.

2. Return database values exactly as stored.

3. Do not add currency symbols unless they exist in the
   database value.

4. Do not change numeric formatting unnecessarily.

5. Do not convert currencies.

6. Do not invent database records.

7. If the requested information does not exist in the
   database, clearly say that the information was not found.

8. Before answering, inspect the database schema and choose
   the appropriate table and columns.

9. For aggregation questions such as count, average, total,
   maximum or minimum, calculate the result using SQL.

10. For questions about a specific order or return, use the
    corresponding order/return record from the database.
"""
)


# ============================================================
# 11. STATE
# ============================================================

class State(TypedDict):

    messages: Annotated[list[BaseMessage],add_messages]
    route: str
    answer: str

# ============================================================
# 12. SUPERVISOR
# ============================================================

def supervisor(state: State):

    messages = state["messages"]

    # Current user question
    question = messages[-1].content

    supervisor_prompt = f"""
You are a supervisor responsible for routing customer
questions to the correct system.

There are two systems:

1. RAG
2. SQL

==================================================
RAG
==================================================

Choose RAG when the user is asking for information contained
in the PDF documents.

Typical RAG questions include:

- What is the refund policy?
- What is the return policy?
- Can I return a product?
- What are the warranty terms?
- How long is the warranty?
- What is the shipping policy?
- What are the conditions for getting a refund?
- What does the product manual say?
- What are the product features mentioned in the documents?

==================================================
SQL
==================================================

Choose SQL when the user is asking about actual data stored
in the database.

The database contains product-related data such as:

- Orders
- Returns
- Products
- Customers
- Order details
- Return details
- Order status
- Return status
- Quantity
- Price
- Amount

Examples:

- Where is my order?
- What is the status of order 1001?
- Has order 1001 been shipped?
- What products did customer John buy?
- How many orders are there?
- Which product has the most orders?
- Which products were returned?
- How many products were returned?
- Show me returned products.
- What is the return status of order 1001?
- Which customer ordered iPhone 15?
- How many iPhone 15 orders are there?
- What is the total sales amount?
- Which product generated the highest sales?

==================================================
IMPORTANT DISTINCTION
==================================================

If the question asks about a POLICY, RULE, PROCEDURE,
DOCUMENTATION or GENERAL PRODUCT INFORMATION:

→ RAG


If the question asks about an ACTUAL ORDER, RETURN,
CUSTOMER, PRODUCT RECORD, QUANTITY, PRICE, STATUS or
TRANSACTION stored in the database:

→ SQL


Examples:

"Can I return this product?"
→ RAG

"Was my product returned?"
→ SQL

"What is the refund policy?"
→ RAG

"Did order 1005 get refunded?"
→ SQL

"How long is the warranty?"
→ RAG

"What is the warranty status of order 1005?"
→ SQL

==================================================
OUTPUT
==================================================

Return ONLY one word:

RAG

or

SQL


User question:

{question}
"""

    response = model.invoke(supervisor_prompt)
    decision = response.content.strip().upper()
    print("Supervisor decision:", decision)

    if "SQL" in decision:
        return {"route": "sql"}

    return {"route": "rag"}

# ============================================================
# 13. RAG NODE
# ============================================================

def rag_node(state: State):
    response = rag_agent.invoke({"messages": state["messages"]})
    answer = response["messages"][-1].content
    return {"messages": [AIMessage(content=answer)],"answer": answer}

# ============================================================
# 14. SQL NODE
# ============================================================

def sql_node(state: State):

    response = sql_agent.invoke({"input": state["messages"][-1].content})
    answer = response["output"]
    return {"messages": [AIMessage(content=answer)],"answer": answer}

# ============================================================
# 15. ROUTER
# ============================================================

def route_question(state: State) -> Literal["rag", "sql"]:

    return state["route"]


# ============================================================
# 16. BUILD LANGGRAPH
# ============================================================

checkpointer = InMemorySaver()
builder = StateGraph(State)

# Nodes
builder.add_node("supervisor",supervisor)
builder.add_node("rag",rag_node)
builder.add_node("sql",sql_node)

# START → Supervisor
builder.add_edge(START,"supervisor")

# Supervisor → RAG / SQL
builder.add_conditional_edges("supervisor",route_question,
    {
        "rag": "rag",
        "sql": "sql"
    }
)


# RAG → END
builder.add_edge("rag",END)

# SQL → END
builder.add_edge("sql",END)

# Compile
graph = builder.compile(checkpointer=checkpointer)

# ============================================================
# 17. CHAT SERVICE
# ============================================================

def chat_service(session_id: str,message: str):
    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke({"messages": [HumanMessage(content=message)]},config=config)
    return result["answer"]
