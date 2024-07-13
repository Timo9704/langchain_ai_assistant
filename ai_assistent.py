from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import Pinecone

# Laden der Umgebungsvariablen
load_dotenv()

# Erstellen des FastAPI-App-Objekts
app = FastAPI()

# Initialisieren des Chat-Modells und anderer Komponenten
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
vectorstore = Pinecone.from_existing_index("aquabot", embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

# Hilfsfunktion, um Dokumente zu formatieren
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Definition der Datenstruktur für Anfragen
class ChatRequest(BaseModel):
    question: str

# API Route zum Empfangen des Prompts und Senden der Antwort
@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        answer = rag_chain.invoke(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
