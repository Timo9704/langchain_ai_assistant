import logging
from fastapi import FastAPI, HTTPException
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain_community.chat_message_histories import ChatMessageHistory

from model.input_model import RequestBody

load_dotenv()

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

# FastAPI config
app = FastAPI()

# LangChain config
llm = ChatOpenAI(model="gpt-4o-mini")

# Vectorstore config
vectorstore = Pinecone.from_existing_index("aquabot", embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
session_store = {}

#This functions are based on the tutorial from the LangChain documentation.
#https://python.langchain.com/v0.2/docs/tutorials/chatbot/

# Helper functions
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@app.post("/assistant/")
async def chat(request: RequestBody):
    try:

        system_prompt = (
            "Du bist ein Aquaristik-Experte für Fragen und Antworten."
            "Nutze für deine Antworten die folgenden Informationen aus deinem Fachgebiet."
            "Wenn du keine Antwort weißt, schreibe einfach 'Entschuldigung, das weiß ich nicht'."
            "Wenn du beleidigt wirst, antworte einfach 'Es tut mir leid, dass ich dir nicht helfen konnte'."
            f"Beachte, dass der Fragesteller sich auf dem Niveau {request.preferences.experience_level} befindet."
            f"Bitte antworte von der Länge und detailtiefe {request.preferences.detail_level}."
            "Weiterhin bekommst du die folgende Informationen über das Aquarium, falls du Fragen zu bestimmten "
            "Problemen mit Bezug auf das Aquarium des Fragestellers beantworten sollst:"
            f"Das Aquarium hat ein Volumen von {request.aquarium_data.aquarium_liter} Litern."
            f"Die Wasserparameter der letzten Messung sind: {request.aquarium_data.water_parameters.json}"
            "\n\n"
            "{context}"
        )

        system_prompt_history = (
            "Gegeben ist ein Chatverlauf und die aktuellste Benutzerfrage, "
            "die sich möglicherweise auf den Kontext des Chatverlaufs bezieht. "
            "Formuliere eine eigenständige Frage, die ohne den Chatverlauf verstanden werden kann. "
            "Beantworte die Frage NICHT, sondern formuliere sie bei Bedarf um oder gebe sie unverändert zurück."
        )

        prompt_with_history = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_history),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, prompt_with_history
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        answer = conversational_rag_chain.invoke(
            {"input": request.ai_input.human_input},
            config={
                "configurable": {"session_id": request.ai_input.session_id}
            },
        )["answer"]
        return {"answer": answer, "session_id": request.ai_input.session_id}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
