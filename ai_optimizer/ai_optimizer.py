import logging
import os

from fastapi import FastAPI, HTTPException
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain_google_community import GoogleSearchAPIWrapper
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.llm_math.base import LLMMathChain
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from model.output_model import AquariumPlanningResult
from model.input_model import RequestBody

load_dotenv()

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

# FastAPI config
app = FastAPI()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
llm_db = ChatOpenAI(model="gpt-4-0613", temperature=0, streaming=True)

parser = JsonOutputParser(pydantic_object=AquariumPlanningResult)

def tool_retriever_vectorstore_general():
    embedding = OpenAIEmbeddings()
    pinecone_index = Pinecone.from_existing_index("aquabot", embedding=embedding)

    def retrieve_knowledge(query: str):
        results = pinecone_index.similarity_search(query, k=8)
        return results

    retriever_tool = Tool(
        name="Wissensdatenbank für Aquaristik und Aquascaping",
        func=retrieve_knowledge,
        description="Eine Wissensdatenbank, die Informationen zu Aquarien, Technik, Fischen und Pflanzen liefert. "
                    "Dort gibt es auch Tipps und Tricks bei Problemen oder Fragen. Es darf jedoch nicht zu allgemein sein, sondern muss ein spezifisches Thema behandeln."
    )
    return retriever_tool

def tool_retriever_vectorstore_fishes():
    embedding = OpenAIEmbeddings()
    pinecone_index = Pinecone.from_existing_index("aiplannerfishes", embedding=embedding)

    def retrieve_knowledge(query: str):
        results = pinecone_index.similarity_search(query, k=8)
        return results

    retriever_tool = Tool(
        name="Wissensdatenbank für Fische",
        func=retrieve_knowledge,
        description="Eine Wissensdatenbank, die Informationen zu einem spezifischen Fisch, wie Namen, Verhalten, wasserwerte und Haltung liefert."
    )
    return retriever_tool

def tool_retriever_vectorstore_plants():
    embedding = OpenAIEmbeddings()
    pinecone_index = Pinecone.from_existing_index("aiplannerplants", embedding=embedding)

    def retrieve_knowledge(query: str):
        results = pinecone_index.similarity_search(query, k=8)
        return results

    retriever_tool = Tool(
        name="Wissensdatenbank für Pflanzen",
        func=retrieve_knowledge,
        description="Eine Wissensdatenbank, die Informationen zu einer spezifischen Pflanze, wie Namen, Lichtbedarf oder CO2-Bedarf liefert."
    )
    return retriever_tool

def tool_math_calculator():
    llm_math_chain_tool = LLMMathChain.from_llm(llm)

    calculator_tool = Tool(
        name="Calculator",
        func=llm_math_chain_tool.run,
        description="Ein Taschenrechner, wenn du mathematische Berechnungen durchführen möchtest."
    )
    return calculator_tool


def tool_google_search_aquarium():
    search = GoogleSearchAPIWrapper(google_cse_id=os.environ.get("GOOGLE_CSE_ID_ALL"))

    google_search_tool = Tool(
        name="Google Search für alle Themen rund um Aquaristik und Aquascaping.",
        description="Eine Websuche. Nützlich, wenn du ganz spezifsches Wissen im Bereich Aquaristik und Aquascaping benötigst.",
        func=search.run
    )
    return google_search_tool

@app.post("/optimizer/")
async def chat(request: RequestBody):
    try:
        tools = [
            tool_math_calculator(),
            tool_retriever_vectorstore_general(),
            tool_retriever_vectorstore_fishes(),
            tool_retriever_vectorstore_plants(),
            #tool_google_search_aquarium()
        ]

        promptTemplate = PromptTemplate.from_template(
            template=f"""
            Du bist ein Aquarium-Experte und analysierst für Aquarianer deren Aquarien. 
            Deine Aufgabe besteht darin, basierend auf den angegebenen Angaben, aktuelle Probleme zu finden und dem Aquarianer Verbesserungen aufzuzeigen.
            Du gehst immer Schritt für Schritt.
            
            Aquarium-Details: {request.aquariumInfo}
            
            Der Aquarianer beschreibt dir, dass er Probleme mit seinem Aquarium hat und du sollst ihm helfen, diese zu lösen.
            Hier sind die Angaben des Aquarianers:
            - Zum Aquarium allgemein: {request.aquariumProblemDescription}
            - Zu den Fischen: {request.fishProblemDescription}
            - Zu den Pflanzen: {request.plantProblemDescription}
            
            Die finale Antwort soll die folgenden Struktur haben:
            - Optimierung des Aquariums:
                - erkannte Probleme:
                - Verbesserungsvorschläge:
            - Optimierung der Technik:
                - erkannte Probleme:
                - Verbesserungsvorschläge:
            - Optimierung des Besatzes:
                - erkannte Probleme:
                - Verbesserungsvorschläge:
            - Optimierung der Pflanzen:
                - erkannte Probleme:
                - Verbesserungsvorschläge:
                
            Wenn es in einem Bereich keine Probleme gibt, gib für diesen Bereich einfach "Keine Probleme gefunden!" zurück.
            """,
        )

        prompt = hub.pull("hwchase17/react")
        react_agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True, maxIterations=2)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        print(answer)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
