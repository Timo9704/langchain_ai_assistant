import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from fastapi import HTTPException
from langchain.chains.llm_math.base import LLMMathChain
from langchain_community.vectorstores import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

from model.output_model import PlantsPlanningResultWrapper
from model.input_model import PlanningData

load_dotenv()

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

# LLM config
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def planning_plants_controller(request: PlanningData):
    start_time = time.time()
    try:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(planning_background_plants, request),
                executor.submit(planning_plants_amount, request)
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
        logger.error(f"Error Plants Controller: {str(e)}")
        return ""


def convert_to_json(answer1, answer2):
    structured_llm = llm.with_structured_output(PlantsPlanningResultWrapper)
    structured_answer = structured_llm.invoke(str(answer1) + " " + str(answer2))
    return structured_answer


def planning_plants_amount(request: PlanningData):
    llm_math_chain_tool = LLMMathChain.from_llm(llm)

    try:
        tools = [
            Tool(
                name="Calculator",
                func=llm_math_chain_tool.run,
                description="Ein Taschenrechner, wenn du mathematische Berechnungen durchführen möchtest."
            ),
        ]

        promptTemplate = PromptTemplate.from_template(
            template=f"""
                Du bist ein Pflanzen-Planer für Aquarien. Deine Aufgabe ist es die geeignete Menge Pflanzen für ein bestehendes Aquarium zu finden.
                Angaben zum Aquarium: {request.aquariumInfo}

                1. Berechne die Grundfläche des Aquariums.
                
                2. Berechne die Anzahl der Gesamtpflanzen für die Grundfläche des Aquariums.
                - Eine Pflanze pro 130 cm² Grundfläche.
                
                3. Berechne die Anzahl für die Vordergrund-, Mittelgrund- und Hintergrundpflanzen.
                - Vordergrund: 1/5 der Gesamtpflanzenanzahl
                - Mittelgrund: 1/5 der Gesamtpflanzenanzahl
                - Hintergrund: 2/5 der Gesamtpflanzenanzahl
                
                Die Antwort muss so aussehen: 'Vordergrundpflanzen: X Stück. Mittelgrundpflanzen: Y Stück. Hintergrundpflanzen: Z Stück.'
                """,
        )
        # ReAct config
        react_prompt = hub.pull("hwchase17/react")
        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        return answer
    except Exception as e:
        logger.error(f"Error Plants Amount Calculation: {str(e)}")
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

        prompt = f"""
            Du bist als Pflanzen-Planer für Aquarien tätig. Deine Aufgabe besteht darin, passende Pflanzen für ein spezifisches Aquarium zu finden, basierend auf den angegebenen Bedingungen.

            Aquarium-Details: {request.aquariumInfo}

            Anleitung zur Auswahl basierend auf Lichtbedarf:
            - Niedrige Beleuchtungsstärke: Wähle ausschließlich Pflanzen für niedrigen Lichtbedarf.
            - Mittlere Beleuchtungsstärke: Wähle Pflanzen für niedrigen und mittleren Lichtbedarf.
            - Hohe Beleuchtungsstärke: Wähle Pflanzen für alle Lichtbedarfsstufen (niedrig, mittel, hoch).

            Anleitung zur Auswahl nach Wuchshöhen:
            - 0-5 cm: Wähle Vordergrundpflanzen.
            - 5-15 cm: Wähle Mittelgrundpflanzen.
            - 15-30 cm: Wähle Hintergrundpflanzen.

            Zusätzliche Auswahlkriterien:
            - {'Bevorzuge eine große Anzahl an sehr kleinen Vordergrundpflanzen.' if request.useForegroundPlants else 'Wähle weniger sehr kleine Vordergrundpflanzen.'}
            - {'Moospflanzen sollen in die Auswahl einbezogen werden.' if request.useMossPlants else 'Moospflanzen dürfen nicht ausgewählt werden.'}
            - Mindestwachstumsgeschwindigkeit: {request.growthRate}.

            Die Auswahl sollte mindestens eine Pflanze für den Vorder-, Mittel- und Hintergrund umfassen und dabei Wachstumsgeschwindigkeit, Lichtbedarf und CO2-Bedarf jeder Pflanze berücksichtigen.
        """

        # Ausführen des RAG Chains
        result = rag_chain.invoke(prompt)
        return result
    except Exception as e:
        logger.error(f"Error Plants Planning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
