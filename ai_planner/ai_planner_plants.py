import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from fastapi import HTTPException
from langchain_community.vectorstores import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

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
                executor.submit(planning_plants, request),
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


class SearchInput(BaseModel):
    json: str = Field(description="Länge des Aquariums ohne Einheit")


def planning_plants_amount(request: PlanningData):
    start_time = time.time()
    @tool("search-tool", args_schema=SearchInput, return_direct=True)
    def calculate_plants_amount(json_string: str):
        """
            Berechnet die Anzahl der benötigten Pflanzen basierend auf der Flächengröße und Prozentsätzen für Vorder-, Mittel- und Hintergrund.
            Diese Methode gibt die Anzahl der Pflanzen für jeden Bereich als formatierten String zurück.
            Die Eingabe erfolgt als JSON-Objekt mit den folgenden Feldern:
            - length: Länge des Aquariums ohne Einheit
            - width: Breite des Aquariums ohne Einheit
            - foreground_percentage: Prozentuale Anteil der Vordergrundpflanzen
            - middle_percentage: Prozentuale Anteil der Mittelgrundpflanzen
            - background_percentage: Prozentuale Anteil der Hintergrundpflanzen
            """

        # Umwandeln des JSON-Strings in ein Python-Dictionary
        data = json.loads(json_string)

        # Zugriff auf die einzelnen Werte
        length = data['length']
        width = data['width']
        foreground_percentage = data['foreground_percentage']
        middle_percentage = data['middle_percentage']
        background_percentage = data['background_percentage']

        # Berechnung der Fläche des Aquariums
        area = length * width

        # Berechnung der Anzahl der Pflanzen für jeden Bereich
        foreground_plants = (area * (foreground_percentage / 100)) / 130
        middle_plants = (area * (middle_percentage / 100)) / 130
        background_plants = (area * (background_percentage / 100)) / 130

        return f"Vordergrundpflanzen: {foreground_plants:.0f} Stück. Mittelgrundpflanzen: {middle_plants:.0f} Stück. Hintergrundpflanzen: {background_plants:.0f} Stück."

    try:
        tools = [
            calculate_plants_amount
        ]

        promptTemplate = PromptTemplate.from_template(
            template=f"""
                Du bist ein Pflanzen-Planer für Aquarien. Deine Aufgabe ist es die geeignete Menge Pflanzen für ein bestehendes Aquarium zu finden.
                Angaben zum Aquarium: {request.aquariumInfo}

                1. Berechne die Anzahl der Pflanzen für den Vorder-, Mittel- und Hintergrund des Aquariums aus.
                Entnehme die Länge und Breite aus den Informationen des gegebenen Aquariums.
                Für die Prozentsätze der Pflanzen in den verschiedenen Bereichen des Aquariums, gehe davon aus, dass die Vordergrundpflanzen 20% der Fläche, die Mittelgrundpflanzen 40% der Fläche und die Hintergrundpflanzen 40% der Fläche einnehmen.
                
                Die Antwort muss so aussehen: 'Vordergrundpflanzen: X Stück. Mittelgrundpflanzen: Y Stück. Hintergrundpflanzen: Z Stück.'
                """,
        )
        # ReAct config
        react_prompt = hub.pull("hwchase17/react")
        react_agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        end_time = time.time()
        logger.info(f"Planning Plants Amount Calculation finished in {end_time - start_time} seconds")
        return answer
    except Exception as e:
        logger.error(f"Error Plants Amount Calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def planning_plants(request: PlanningData):
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

            Die Auswahl sollte immer zwei Pflanzen für den Vorder-, Mittel- und Hintergrund umfassen und dabei Wachstumsgeschwindigkeit, Lichtbedarf und CO2-Bedarf jeder Pflanze berücksichtigen.
        """

        result = rag_chain.invoke(prompt)
        return result
    except Exception as e:
        logger.error(f"Error Plants Planning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
