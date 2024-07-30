import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from fastapi import FastAPI, HTTPException
from langchain_community.vectorstores import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

from model.output_model import PlantsPlanningResult
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

# ReAct config
react_prompt = hub.pull("hwchase17/react")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def planning_plants_controller(request: PlanningData):
    start_time = time.time()
    try:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(planning_background_plants, request)
            ]

            answers = []
            for future in as_completed(futures):
                answers.append(future.result())

        if request.planningMode == "Pflanzen":
            structured_answer = convert_to_json(*answers)
        else:
            structured_answer = answers
        end_time = time.time()
        logger.info(f"Planning planning_plants_controller finished in {end_time - start_time} seconds")
        return structured_answer
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return ""


def convert_to_json(answer1):
    structured_llm = llm.with_structured_output(PlantsPlanningResult)
    structured_answer = structured_llm.invoke(answer1)
    return structured_answer


def planning_foreground_plants(request: PlanningData):
    def tool_retriever_vectorstore():
        embedding = OpenAIEmbeddings()
        pinecone_index = Pinecone.from_existing_index("aiplannerplants", embedding=embedding)

        def retrieve_knowledge(query: str):
            results = pinecone_index.similarity_search(query, k=3)
            return results

        retriever_tool = Tool(
            name="Knowledge retriever für Pflanzen",
            func=retrieve_knowledge,
            description="Ein Knowledge-Retriever, der Informationen zu Pflanzen liefert. "
        )
        return retriever_tool

    try:
        tools = [
            tool_retriever_vectorstore(),
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
                   - **Bedingungen**:
                       - Limitiere die Anzahl der Pflanzen auf 3.
                       
                Die Antwort ist eine unterteilte Liste in Deutsch mit Name der Pflanze, Wachstumsgeschwindigkeit, Lichtbedarf und CO2-Bedarf der einzelnen Pflanzen.
                Wenn du keine Pflanzen findest, schreibe 'Keine Pflanzen gefunden!'.
                """,
        )

        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        return answer
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def planning_midground_plants(request: PlanningData):
    def tool_retriever_vectorstore():
        embedding = OpenAIEmbeddings()
        pinecone_index = Pinecone.from_existing_index("aiplannerplants", embedding=embedding)

        def retrieve_knowledge(query: str):
            results = pinecone_index.similarity_search(query, k=3)
            return results

        retriever_tool = Tool(
            name="Knowledge retriever für Pflanzen",
            func=retrieve_knowledge,
            description="Ein Knowledge-Retriever, der Informationen zu Pflanzen liefert. "
        )
        return retriever_tool

    try:
        tools = [
            tool_retriever_vectorstore(),
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
                   - **Bedingungen**:
                       - Limitiere die Anzahl der Pflanzen auf 3 Vordergrund-, 3 Mittelgrund- und 3 Hintergrundpflanzen.
                       
                Die Antwort ist eine unterteilte Liste in Deutsch mit Name der Pflanze, Wachstumsgeschwindigkeit, Lichtbedarf und CO2-Bedarf der einzelnen Pflanzen.
                Wenn du keine Pflanzen findest, schreibe 'Keine Pflanzen gefunden!'.
                """,
        )

        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        return answer
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def planning_background_plants(request: PlanningData):
    try:
        vectorstore = Pinecone.from_existing_index("aiplannerplants", embedding=OpenAIEmbeddings())
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
            Du bist ein Pflanzen-Planer für Aquarien. Deine Aufgabe ist es, geeignete Pflanzen für ein bestehendes Aquarium zu finden.
            Dies sind Angaben zum bestehenden Aquarium: {request.aquariumInfo}

            Nutze das folgende Mapping für den Lichtbedarf:
            1. niedriger Beleuchtungsstärke -> nur Pflanzen für niedrigen Lichtbedarf
            2. mittlerer Beleuchtungsstärke -> Pflanzen für niedrigen und mittleren Lichtbedarf
            3. hoher Beleuchtungsstärke -> Pflanzen für niedrigen, mittleren und hohen Lichtbedarf

            Die Auswahl sollte Pflanzen für den Vordergrund, Mittelgrund und Hintergrund umfassen. Berücksichtige die Wachstumsgeschwindigkeit, den Lichtbedarf und den CO2-Bedarf jeder Pflanze.
        """

        # Ausführen des RAG Chains
        result = rag_chain.invoke(prompt)

        return result
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
