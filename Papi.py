from fastapi import FastAPI
from pydantic import BaseModel
from Pservice2 import chat_service

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
def chat(data: ChatRequest):
    answer = chat_service(data.session_id,data.message)
    return {"answer": answer}
