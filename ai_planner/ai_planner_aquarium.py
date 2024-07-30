import asyncio
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from fastapi import FastAPI, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
    logger.error(f"Planning Aquarium took {end_time - start_time} seconds")

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(planning_animals_controller, request),
            executor.submit(planning_plants_controller, request)
        ]

        for future in as_completed(futures):
            answers.append(future.result())

    structured_answer = convert_to_json(*answers)
    return structured_answer


def convert_to_json(answer1, answer2, answer3):
    structured_llm = llm.with_structured_output(AquariumPlanningResult)
    structured_answer = structured_llm.invoke(str(answer1) + " " + str(answer2) + " " + str(answer3))
    return structured_answer


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def planning_aquarium(request: PlanningData):
    try:
        vectorstore = Pinecone.from_existing_index("aiplanneraquarium", embedding=OpenAIEmbeddings())
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
            Du bist ein Planer für die Auswahl eines optimalen Aquariums für einen Kunden. 
                Deine Aufgabe ist es, ein geeignete Aquarien auf Basis von Anforderungen auszuwählen.
                
                Dies sind die Anforderungen des Kunden:
                Das Aquarium muss weniger als {request.availableSpace} cm lang sein.
                Das Aquarium muss weniger als {request.maxVolume} Liter fassen, aber nicht weniger als 54 Liter.
                Das Aquarium muss weniger als {request.maxCost} Euro kosten.
                Das Aquarium muss {'mit' if request.needCabinet else 'ohne'} Unterschrank sein.
                Das Aquarium {'muss ein' if request.isSet else 'darf auf keinen Fall ein'} Set mit Filter, Beleuchtung und Heizer sein.
                Wenn mehrere passende Aquarien gefunden werden, wähle das größere aus.
                
                Deine Antwort ist eine Liste mit Name des Aquariums, alle Eigenschaften und Produkten auf Deutsch!'.
        """

        result = rag_chain.invoke(prompt)
        print(result)
        return result

    except Exception as e:
        logger.error(f"Error: {str(e)}")
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
        # ReAct config
        react_prompt = hub.pull("hwchase17/react")
        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True, maxIterations=2)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        end_time = time.time()
        logger.error(f"Planning Technik took {end_time - start_time} seconds")
        return answer
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
