import asyncio
import logging
import os

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
    results = search_aquarium.results(query, 2)
    return results


async def planning_aquarium_controller(request: PlanningData):
    result1_task = asyncio.create_task(planning_aquarium(request))
    result2_task = asyncio.create_task(planning_animals_controller(request))
    result3_task = asyncio.create_task(planning_plants_controller(request))

    answer1 = await result1_task
    answer2 = await result2_task
    answer3 = await result3_task

    structured_answer = convert_to_json(answer1, str(answer2), str(answer3))
    return structured_answer


def convert_to_json(answer1, answer2, answer3):
    structured_llm = llm.with_structured_output(AquariumPlanningResult)
    structured_answer = structured_llm.invoke(answer1 + " " + answer2 + " " + answer3)
    return structured_answer


async def planning_aquarium(request: PlanningData):
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
                name="Google Suche für Links zu Aquarien",
                description="Eine Websuche, um Links zu Aquarien zu bekommen.",
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


def planning_midground_plants(request: PlanningData):
    try:
        tools = [
            Tool(
                name="SQL Database",
                func=db_chain_tool.run,
                description="Eine SQL-Datenbank App-DB, wenn du nach Pflanzen in der Datenbank suchen sollst."
            ),
            Tool(
                name="Calculator",
                func=llm_math_chain_tool.run,
                description="Ein Taschenrechner, wenn du mathematische Berechnungen durchführen möchtest."
            ),
            Tool(
                name="Google Suche für Links zu Aquarienpflanzen",
                description="Eine Websuche, um Links zu Planzen zu bekommen.",
                func=top2_results_aquarium,
            ),
        ]

        promptTemplate = PromptTemplate.from_template(
            template=f"""
               Du bist ein Pflanzen-Planer für Aquarien. Deine Aufgabe ist es, geeignete Pflanzen für ein bestehendes Aquarium zu finden.
                Deine Aufgabe ist es nun, geeignete Pflanzen für das Aquarium zu finden, die den Bedingungen entsprechen.
                Dies sind Angaben zum bestehenden Aquarium: {request.aquariumInfo}

                Wende folgendes Mapping an:
                1. niedrig -> niedrig
                2. mittel -> mittel + niedrig
                3. hoch -> hoch + mittel + niedrig

                1. **Auswahl geeigneter Pflanzen für das Aquarium**:
                   - **Datenbankabfrage**: Suche in der Tabelle 'plants' der App-DB.
                   - **Bedingungen**:
                       - type: Mittelgrund.
                       - co2_demand: Wenn keine CO2-Anlage vorhanden ist, dann dürfen die Pflanzen nur einen niedrigen CO2-Bedarf haben.
                       - light_demand: Lichtbedarf ist "mittel" und darunter.
                       - growth_rate: maximale Wuchsschnelligkeit ist "hoch" und darunter.
                       - Limitiere die Anzahl der Pflanzen auf 3.
                Die Antwort ist eine unterteilte Liste in Deutsch. Wenn du keine Pflanzen findest, schreibe 'Keine Pflanzen gefunden!'.
                """,
        )

        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True, maxIterations=2)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        return answer
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def planning_background_plants(request: PlanningData):
    try:
        tools = [
            Tool(
                name="SQL Database",
                func=db_chain_tool.run,
                description="Eine SQL-Datenbank App-DB, wenn du nach Pflanzen in der Datenbank suchen sollst."
            ),
            Tool(
                name="Calculator",
                func=llm_math_chain_tool.run,
                description="Ein Taschenrechner, wenn du mathematische Berechnungen durchführen möchtest."
            ),
            Tool(
                name="Google Suche für Links zu Aquarienpflanzen",
                description="Eine Websuche, um Links zu Planzen zu bekommen.",
                func=top2_results_aquarium,
            ),
        ]

        promptTemplate = PromptTemplate.from_template(
            template=f"""
                Du bist ein Pflanzen-Planer für Aquarien. Deine Aufgabe ist es, geeignete Pflanzen für ein bestehendes Aquarium zu finden.
                Deine Aufgabe ist es nun, geeignete Pflanzen für das Aquarium zu finden, die den Bedingungen entsprechen.
                Dies sind Angaben zum bestehenden Aquarium: {request.aquariumInfo}

                Wende folgendes Mapping an:
                1. niedrig -> niedrig
                2. mittel -> mittel + niedrig
                3. hoch -> hoch + mittel + niedrig

                1. **Auswahl geeigneter Pflanzen für das Aquarium**:
                   - **Datenbankabfrage**: Suche in der Tabelle 'plants' der App-DB.
                   - **Bedingungen**:
                       - type: Hintergrund.
                       - co2_demand: Wenn keine CO2-Anlage vorhanden ist, dann dürfen die Pflanzen nur einen niedrigen CO2-Bedarf haben.
                       - light_demand: Lichtbedarf ist "mittel" und darunter.
                       - growth_rate: maximale Wuchsschnelligkeit ist "hoch" und darunter.
                       - Limitiere die Anzahl der Pflanzen auf 3.
                Die Antwort ist eine unterteilte Liste in Deutsch. Wenn du keine Pflanzen findest, schreibe 'Keine Pflanzen gefunden!'.
                """,
        )

        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True, maxIterations=2)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        return answer
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))