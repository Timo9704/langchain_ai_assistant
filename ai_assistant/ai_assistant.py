import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import Pinecone

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

app = FastAPI()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
vectorstore = Pinecone.from_existing_index("aquabot", embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RequestData(BaseModel):
    question: str


@app.post("/assistant/")
async def chat(request: RequestData):
    logger.info(f"Received question: {request.question}")
    try:
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
        answer = rag_chain.invoke(request.question)
        logger.info(f"Answer: {answer}")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
