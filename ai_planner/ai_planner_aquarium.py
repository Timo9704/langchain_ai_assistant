import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from fastapi import HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain_google_community import GoogleSearchAPIWrapper
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

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


def planning_aquarium(request: PlanningData):
    try:
        vectorstore = Pinecone.from_existing_index("aiplanneraquarium", embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever(search_kwargs={"k": 16})

        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        prompt = f"""
            Du bist ein Planer für die Auswahl eines optimalen Aquariums für einen Kunden. 
                Deine Aufgabe ist es, ein geeignete Aquarien auf Basis von Anforderungen auszuwählen.
                
                Dies sind die Anforderungen des Kunden:
                Das Aquarium muss weniger als {request.availableSpace} cm lang sein.
                Das Aquarium muss zwischen {request.minVolume} als {request.maxVolume} Liter fassen.
                Das Aquarium muss weniger als {request.maxCost} Euro kosten.
                Das Aquarium muss {'mit' if request.needCabinet else 'ohne'} Unterschrank sein.
                Das Aquarium {'muss ein' if request.isSet else 'darf auf keinen Fall ein'} Set mit Filter, Beleuchtung und Heizer sein.
                
                Deine Antwort ist eine Liste mit Name des Aquariums, alle Eigenschaften und Produkten auf Deutsch!'.
        """

        result = rag_chain.invoke(prompt)
        return result

    except Exception as e:
        logger.error(f"Error Planning Aquarium: {str(e)}")
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
