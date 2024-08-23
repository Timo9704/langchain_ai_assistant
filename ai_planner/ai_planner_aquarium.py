import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain_google_community import GoogleSearchAPIWrapper
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool, tool
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from ai_planner_animals import planning_animals_controller
from ai_planner_plants import planning_plants_controller
from model.output_model import AquariumPlanningResult
from model.input_model import PlanningData

load_dotenv()

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

# LLM config
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

search_aquarium = GoogleSearchAPIWrapper(google_cse_id=os.environ.get("GOOGLE_CSE_ID_AQUARIUM"))


def results_tech(query):
    results = search_aquarium.run(query)
    return results


async def planning_aquarium_controller(request: PlanningData):
    start_time = time.time()
    answers = []
    answer1 = planning_aquarium(request)
    request.aquariumInfo = answer1
    answers.append(answer1)
    end_time = time.time()
    logger.info(f"Planning Aquarium took {end_time - start_time} seconds")

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
    structured_answer = structured_llm.invoke(str(answer1) + " " + str(answer2) + " " + str(answer3) + str(answer4))
    return structured_answer


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def tool_retriever_vectorstore_aquarium():
    embedding = OpenAIEmbeddings()
    pinecone_index = Pinecone.from_existing_index("aiplanneraquarium", embedding=embedding)

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


class SearchInput(BaseModel):
    json: str = Field(description="Länge des Aquariums ohne Einheit")


def planning_aquarium(request: PlanningData):
    try:
        start_time = time.time()

        tools = [
            tool_retriever_vectorstore_aquarium(),
        ]

        promptTemplate = PromptTemplate.from_template(f"""
            Du bist ein Planer für die Auswahl eines optimalen Aquariums für einen Kunden. 
                Deine Aufgabe ist es, ein geeignetes Aquarium auf Basis von Anforderungen auszuwählen.
                
                Suche ein Aquarium auf Basis der folgenden Anforderungen:
                Länge: maximal {request.availableSpace} cm.
                Volumen: zwischen {request.minVolume} bis {request.maxVolume} Liter.
                Preis: maximal {request.maxCost} Euro.
                Unterschrank: {'ja' if request.needCabinet else 'nein'}.
                als Set: {'ja' if request.isSet else 'nein'}.
                                   
                Deine Antwort ist eine ausführliche Liste mit allen Daten des Aquarium, wie z.B. 
                Name des Aquariums, alle Eigenschaften und Produkten auf Deutsch für das Aquarium!'.
        """),

        # ReAct config
        react_prompt = hub.pull("hwchase17/react")
        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        end_time = time.time()
        logger.info(answer)
        logger.info(f"Planning Aquarium finished in {end_time - start_time} seconds")
        return answer
    except Exception as e:
        logger.error(f"Error Aquarium Planning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def planning_tech(request: PlanningData):
    start_time = time.time()
    try:
        tools = [
            Tool(
                name="Google Suche für Links zu Technikproukten",
                description="Eine Websuche, um Technik oder Sets zu suchen.",
                func=results_tech,
            ),
        ]

        promptTemplate = PromptTemplate.from_template(
            template=f"""
                Als Planer für Aquarien-Technik prüfst du zunächst, ob es sich bei dem gegebenen Aquarium um ein Set-Aquarium handelt, das bereits mit den notwendigen technischen Komponenten ausgestattet ist.

                Aquarium-Details: {request.aquariumInfo}

                Aufgabe:
                - Ist das Aquarium ein Set-Aquarium mit bereits integriertem Filter, Heizer und Beleuchtung? Falls ja, sind keine weiteren Technikprodukte erforderlich.
                - Ist das Aquarium ein Basis-Aquarium ohne technische Ausstattung, führe die folgenden Schritte durch:
                    - Filter: Suche nach einem geeigneten Filter, der der Literanzahl des Aquariums entspricht.
                    - Heizer: Suche nach einem Heizer, der zur Literanzahl des Aquariums passt.
                    - Beleuchtung: Suche nach einer Beleuchtung, die der Länge des Aquariums gerecht wird.

                Ergebnis:
                Erstelle eine Liste mit den empfohlenen Produkten für Filter, Heizer und Beleuchtung, falls notwendig. Die Informationen sollten in deutscher Sprache aufgelistet werden.
            """
        )
        # ReAct config
        react_prompt = hub.pull("hwchase17/react")
        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True, maxIterations=2)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        end_time = time.time()
        logger.info(f"Planning Technik took {end_time - start_time} seconds")
        return answer
    except Exception as e:
        logger.error(f"Error Planning Technic: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
