import logging
from multiprocessing import Pool

from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.llm_math.base import LLMMathChain
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

from model.output_model import FishesPlanningResult
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

llm_math_chain_tool = LLMMathChain.from_llm(llm)

# ReAct config
react_prompt = hub.pull("hwchase17/react")


def planning_animals_controller(request: PlanningData):
    pool = Pool()
    result1 = pool.apply_async(planning_fishes, [request])
    answer1 = result1.get()
    pool.close()

    if request.planningMode == "Besatz":
        structured_answer = convert_to_json(answer1)
    else:
        structured_answer = answer1

    return structured_answer


def convert_to_json(answer1):
    structured_llm = llm.with_structured_output(FishesPlanningResult)
    structured_answer = structured_llm.invoke(answer1)
    return structured_answer


def planning_fishes(request: PlanningData):
    try:
        tools = [
            Tool(
                name="SQL Database App-DB",
                func=db_chain_tool.run,
                description="Eine SQL-Datenbank App-DB, wenn du nach Fischen in der Datenbank suchen sollst."
            ),
            Tool(
                name="Calculator",
                func=llm_math_chain_tool.run,
                description="Ein Taschenrechner, wenn du mathematische Berechnungen durchführen möchtest."
            )
        ]

        promptTemplate = PromptTemplate.from_template(
            template=f"""
                Du bist ein Planer für die Auswahl von Fischen, Garnelen und Schnecken für Aquarien. 
                Deine Aufgabe ist es, geeignete Tiere für ein bestehendes Aquarium zu finden.
                Dies sind Angaben zum bestehenden Aquarium: {request.aquariumInfo}

                1. **Auswahl geeigneter Fische für das Aquarium**:
                   - **Datenbankabfrage**: Suche in der Tabelle 'fish' der App-DB.
                   - **Bedingungen**:
                       - pH: der gegebene Wert liegt zwischen min_pH und  max_pH
                       - GH: der gegebene Wert liegt zwischen min_GH und  max_GH
                       - KH: der gegebene Wert liegt zwischen min_KH und  max_KH
                       - liters: der gegebene Wert liegt zwischen min_liters und max_liters.
                       - Limitiere die Anzahl der Fische auf 5.
                Die Antwort ist eine unterteilte Liste in Deutsch. Wenn du keine Fische findest, schreibe 'Keine Fische gefunden!'.
                """,
        )

        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True, maxIterations=2)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        return answer
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))