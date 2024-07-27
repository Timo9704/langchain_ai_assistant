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
from model.output_model import AquariumPlanningResult, PlanningDataLink
from model.input_model import PlanningDataNoLink

load_dotenv()

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

# FastAPI config
app = FastAPI()

# LLM config
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ReAct config
react_prompt = hub.pull("hwchase17/react")

search_aquarium = GoogleSearchAPIWrapper(google_cse_id=os.environ.get("GOOGLE_CSE_ID_AQUARIUM"))


def top2_results_aquarium(query):
    results = search_aquarium.results(query, 2)
    return results


async def search_links_controller(request: PlanningDataNoLink):
    answer = await search_links(request)
    return answer


async def search_links(request: PlanningDataNoLink):
    try:
        tools = [
            Tool(
                name="Google Suche für Links zu Aquarien",
                description="Eine Websuche, um Links zu Aquarien zu bekommen.",
                func=top2_results_aquarium,
            ),
        ]

        promptTemplate = PromptTemplate.from_template(
            template=f"""
                Suche zu jedem Eintrag nach einem Link und füge ihn hinzu.
                Wenn du keinen Link findest, schreibe 'kein Link gefunden'.
                {request}
                """,
        )

        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True, maxIterations=2)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        structured_llm = llm.with_structured_output(PlanningDataLink)
        structured_answer = structured_llm.invoke(answer)
        return structured_answer
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
