import logging
import os
import time

from fastapi import FastAPI, HTTPException
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain_google_community import GoogleSearchAPIWrapper
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.llm_math.base import LLMMathChain
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from model.input_model import RequestBody

load_dotenv()

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

# FastAPI config
app = FastAPI()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


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
        name="Eine Google Websuche, falls du sehr spezifische Fragen hast und keine gute Antwort aus anderen Tools bekommen hast.",
        description="Eine Websuche.",
        func=search.run
    )
    return google_search_tool


def optimize_aquarium(request: RequestBody):
    time_start = time.time()
    try:
        tools = [
            tool_retriever_vectorstore_general(),
            tool_retriever_vectorstore_fishes(),
            tool_retriever_vectorstore_plants(),
        ]

        promptTemplate = PromptTemplate.from_template(
            template=f"""
            Du bist ein Aquarium-Experte und analysierst für Aquarianer deren Aquarien, um ihnen zu helfen, optimale Bedingungen zu schaffen. 
            Gehe Schritt für Schritt vor und nutze alle verfügbaren Informationen, um eine umfassende Beratung zu bieten.

            Aquarium-Details: {request.aquariumInfo} {request.aquariumTechInfo} 
            Problembeschreibung des Aquarianers: 
            {request.aquariumProblemDescription}
            {'Das Wasser hat keine Trübung oder Verfärbung.' if request.waterClear else 'Das Wasser ist' + request.waterTurbidity}
            
            1. Ermittle Probleme:
            - Erkenne zuerst die Probleme aus der Beschreibung des Aquarianers.
            - Hat der Filter ausreichend Durchfluss?
            - Ist der Heizer für die Beckengröße geeignet?
            Beachte, dass nicht alle Informationen relevant sein müssen.
            
            2. Erarbeite Lösungsvorschläge für jedes erkannte Problem.

            - Falls kein Problem vorliegt, sollte klar "Keine Probleme gefunden!" stehen.
            """,
        )

        prompt = hub.pull("hwchase17/react")
        react_agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        end_time = time.time()
        logger.info(f"Optimize Aquarium took {end_time - time_start} seconds")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
