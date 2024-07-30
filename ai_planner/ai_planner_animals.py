import logging
import time

from fastapi import HTTPException
from langchain_community.vectorstores import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain import hub

from concurrent.futures import ProcessPoolExecutor, as_completed
from model.output_model import FishesPlanningResult
from model.input_model import PlanningData

load_dotenv()

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)


# LLM config
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def planning_animals_controller(request: PlanningData):
    start_time = time.time()
    try:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(planning_fishes, request),
            ]

            answers = []
            for future in as_completed(futures):
                answers.append(future.result())

        if request.planningMode == "Besatz":
            structured_answer = convert_to_json(*answers)
        else:
            structured_answer = answers
        end_time = time.time()
        logger.info(f"Planning planning_animals_controller finished in {end_time - start_time} seconds")
        return structured_answer
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return ""


def convert_to_json(answer1):
    structured_llm = llm.with_structured_output(FishesPlanningResult)
    structured_answer = structured_llm.invoke(answer1)
    return structured_answer

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def planning_fishes(request: PlanningData):
    try:
        vectorstore = Pinecone.from_existing_index("aiplannerfishes", embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        # Definieren des Prompts, um mit dem RAG Chain zu interagieren
        prompt = f"""
            Du bist ein Planer für die Auswahl von Fischen, Garnelen und Schnecken für Aquarien. 
                Deine Aufgabe ist es, geeignete Tiere für ein bestehendes Aquarium zu finden.
                Dies sind Angaben zum bestehenden Aquarium: {request.aquariumInfo}
                Die Wasserwerte sind: {request.waterValues}

                1. **Auswahl geeigneter Fische für das Aquarium und die gegebenen Wasserwerte**:
                   - **Bedingungen**:
                       - Die Fische müssen sich mit den gegebenen Wasserwerten vertragen.
                       - Die Fische dürfen auf gar keinen Fall mehr Platz beanspruchen, als das Aquarium bietet.
                       - Begrenze die Anzahl der Fische auf 5.
                       
                Die Antwort ist eine unterteilte Liste in Deutsch mit Name des Fisches, Temperatur, pH-Wert, GH-Wert, KH-Wert und Beckengröße der einzelnen Fische.
                Wenn du keine Fische findest, schreibe 'Keine Fische gefunden!'.
        """

        result = rag_chain.invoke(prompt)
        return result
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))