import asyncio
import logging
import os

from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain_google_community import GoogleSearchAPIWrapper
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

from model.output_model import PlanningDataLink
from model.input_model import PlanningDataNoLink, AquariumDataNoLink, FishDataNoLink, PlantDataNoLink

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
    result1_task = asyncio.create_task(search_links(request.aquarium))
    result2_task = asyncio.create_task(search_links(request.fishes))
    result3_task = asyncio.create_task(search_links(request.plants))

    answer1 = await result1_task
    answer2 = await result2_task
    answer3 = await result3_task

    structured_answer = convert_to_json(answer1, answer2, answer3)
    return structured_answer


def convert_to_json(answer1, answer2, answer3):
    structured_llm = llm.with_structured_output(PlanningDataLink)
    structured_answer = structured_llm.invoke(answer1 + " " + answer2 + " " + answer3)
    return structured_answer


async def search_links(items):
    try:
        tools = [
            Tool(
                name="Google Suche für Links zu Aquarien",
                description="Eine Websuche, um Links zu Aquarien zu bekommen.",
                func=top2_results_aquarium,
            ),
        ]

        if isinstance(items, list):
            names = ', '.join(
                [getattr(item, 'fish_lat_name', None) or getattr(item, 'plant_name', None) for item in items])
        else:
            if isinstance(items, AquariumDataNoLink):
                names = items.aquarium_name
            elif isinstance(items, FishDataNoLink):
                names = items.fish_lat_name
            elif isinstance(items, PlantDataNoLink):
                names = items.plant_name
            else:
                names = "Unknown item type"

        promptTemplate = PromptTemplate.from_template(
            template=f"""
                Suche zu jedem Eintrag nach einem Link und füge ihn hinzu.
                Die Ausgabe sollte wie folgt aussehen: 'Name: XYZ, Link: https://www.beispiel.de'.
                Wenn du keinen Link findest, schreibe 'kein Link gefunden'.
                {names}
                """,
        )

        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True, maxIterations=2)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        return answer
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
