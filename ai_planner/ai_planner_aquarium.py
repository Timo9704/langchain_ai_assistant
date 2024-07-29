import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

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

from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

from ai_planner_animals import planning_animals_controller
from ai_planner_plants import planning_plants_controller
from model.output_model import AquariumPlanningResult
from model.input_model import PlanningData

load_dotenv()

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

# FastAPI config
app = FastAPI()

# LLM config
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_db = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# SQL config
db_url = "sqlite:///app.db"
db = SQLDatabase(create_engine(db_url))
db_chain_tool = SQLDatabaseChain.from_llm(llm_db, db, return_direct=True)

# Pinecone config
embedding = OpenAIEmbeddings()
pinecone_index = Pinecone.from_existing_index("aquabot", embedding=embedding)
llm_math_chain_tool = LLMMathChain.from_llm(llm)

# ReAct config
react_prompt = hub.pull("hwchase17/react")


def retrieve_knowledge(query: str):
    results = pinecone_index.similarity_search(query, k=8)
    return results


search_aquarium = GoogleSearchAPIWrapper(google_cse_id=os.environ.get("GOOGLE_CSE_ID_AQUARIUM"))


def top2_results_aquarium(query):
    results = search_aquarium.run(query)
    return results


def results_tech(query):
    results = search_aquarium.run(query)
    return results


async def planning_aquarium_controller(request: PlanningData):
    answers = []
    answer1 = planning_aquarium(request)
    request.aquariumInfo = answer1
    answers.append(answer1)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(planning_tech, request),
            executor.submit(planning_animals_controller, request),
            executor.submit(planning_plants_controller, request)
        ]

        for future in as_completed(futures):
            answers.append(future.result())

    structured_answer = convert_to_json(*answers)
    return structured_answer


def convert_to_json(answer1, answer2, answer3, answer4):
    structured_llm = llm.with_structured_output(AquariumPlanningResult)
    structured_answer = structured_llm.invoke(str(answer1) + " " + str(answer2) + " " + str(answer3) + " " + str(answer4))
    return structured_answer


def planning_aquarium(request: PlanningData):
    try:
        tools = [
            Tool(
                name="SQL Database App-DB",
                func=db_chain_tool.run,
                description="Eine SQL-Datenbank App-DB, wenn du nach Aquarien in der Datenbank suchen sollst."
            ),
            Tool(
                name="Calculator",
                func=llm_math_chain_tool.run,
                description="Ein Taschenrechner, wenn du mathematische Berechnungen durchführen möchtest."
            ),
            Tool(
                name="Google Suche zu Aquarien",
                description="Eine Websuche, um Infomationen zu Aquarien zu bekommen.",
                func=top2_results_aquarium,
            ),
        ]

        promptTemplate = PromptTemplate.from_template(
            template=f"""
                Du bist ein Planer für die Auswahl eines optimalen Aquariums für einen Kunden. 
                Deine Aufgabe ist es, ein geeignete Aquarien auf Basis von Anforderungen auszuwählen.

                1. **Auswahl geeigneter Fische für das Aquarium**:
                   - **Datenbankabfrage**: Suche in der Tabelle 'aquarium' der App-DB.
                   - **Bedingungen**:
                       - length: Weniger als oder gleich {request.availableSpace} cm
                       - volume: Weniger als oder gleich {request.maxVolume} Liter, aber nicht weniger als 54 Liter.
                       - price: Weniger als oder gleich {request.maxCost}
                       - Wenn mehrere passende Aquarien gefunden werden, wähle das größere aus.
                2. **Überprüfung des Preises**:
                     - Suche nach dem aktuellen Preis des ausgewählten Aquariums in der Google Suche.
                Deine Antwort ist das Aquarium mit allen vorhandenen Informationen auf Deutsch!'.
                """,
        )

        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True, maxIterations=2)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        return answer
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def planning_tech(request: PlanningData):
    try:
        tools = [
            Tool(
                name="Calculator",
                func=llm_math_chain_tool.run,
                description="Ein Taschenrechner, wenn du mathematische Berechnungen durchführen möchtest."
            ),
            Tool(
                name="Google Suche für Links zu Technikproukten",
                description="Eine Websuche, um Technik oder Sets zu suchen.",
                func=results_tech,
            ),
        ]

        promptTemplate = PromptTemplate.from_template(
            template=f"""
                Du bist ein Planer für die Auswahl der optimalen Technik für ein Aquariums. 
                Deine Aufgabe ist es vorher zu prüfen, ob das Aquarium ein Set-Aquarium ist oder nicht.
                
                Dies sind Angaben zum bestehenden Aquarium: {request.aquariumInfo}
                
                1. **Prüfe ob das Aquarium ein Set-Aquarium ist**:
                    - Suche in der Google Suche nach dem Namen des Aquariums.
                      Es ist ein Set-Aquarium, wenn Hinweise gibt, dass ein Filter, Heizer und Beleuchtung dabei sind.
                      Wenn es ein Set-Aquarium ist, dann schreibe für jedes Produkt, welches im Set enthalten ist: 'im Set enthalten', aber schreibe auch den Modellnamen auf. Überspringe den zweiten Schritt.
                      Wenn es kein Set-Aquarium ist, dann gehe zum zweiten Schritt.
                  
                2. **Auswahl geeigneter Technik für das Aquarium**:
                    - **Bedingungen**:
                        - das Aquarium ist kein Set-Aquarium
                        - Suche in der Google Suche nach einem Filter auf Aquariumfilter in Kombination mit der Literanzahl des Aquariums.
                        - Suche in der Google Suche nach einem Heizer auf Aquariumheizer in Kombination mit der Literanzahl des Aquariums.
                        - Suche in der Google Suche nach einer Beleuchtung auf Aquariumbeleuchtung in Kombination mit der Literanzahl des Aquariums.
                    
                Die Antwort ist eine unterteilte Liste in Deutsch.
                """,
        )

        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True, maxIterations=2)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        return answer
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
