import logging
import os

from fastapi import FastAPI, HTTPException
from langchain_core.output_parsers import JsonOutputParser
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

from model.output_model import AquariumPlanningResult
from model.input_model import RequestBody

load_dotenv()

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

# FastAPI config
app = FastAPI()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
llm_db = ChatOpenAI(model="gpt-4-0613", temperature=0, streaming=True)

db_url = "sqlite:///aquarium.db"
db = SQLDatabase(create_engine(db_url))

db_chain_tool = SQLDatabaseChain.from_llm(llm_db, db, verbose=True, return_direct=True)

parser = JsonOutputParser(pydantic_object=AquariumPlanningResult)

def tool_retriever_vectorstore():
    embedding = OpenAIEmbeddings()
    pinecone_index = Pinecone.from_existing_index("aquabot", embedding=embedding)

    def retrieve_knowledge(query: str):
        results = pinecone_index.similarity_search(query, k=8)
        return results

    retriever_tool = Tool(
        name="Knowledge retriever",
        func=retrieve_knowledge,
        description="Ein Knowledge-Retriever, der Informationen zu Aquarien, Technik, Fischen und Pflanzen liefert. "
                    "Nutze dieses Tool nur, wenn du keine anderen Informationen hast."
    )
    return retriever_tool


def tool_math_calculator():
    llm_math_chain_tool = LLMMathChain.from_llm(llm)

    calculator_tool = Tool(
        name="Calculator",
        func=llm_math_chain_tool.run,
        description="Ein Taschenrechner, wenn du mathematische Berechnungen durchführen möchtest."
    )
    return calculator_tool


def tool_google_search_aquarium():
    search = GoogleSearchAPIWrapper(google_cse_id=os.environ.get("GOOGLE_CSE_ID_ALL"))

    google_search_tool = Tool(
        name="Google Search",
        description="Eine Websuche. Nützlich, wenn du nach Zubehör, Technik oder anderen Produkten suchen musst.",
        func=search.run
    )
    return google_search_tool

@app.post("/planner/")
async def chat(request: RequestBody):
    try:
        tools = [
            Tool(
                name="SQL Database",
                func=db_chain_tool.run,
                description="Eine SQL-Datenbank die alle verfügbaren Aquarien und Fische enthält."
            ),
            tool_math_calculator(),
            tool_retriever_vectorstore(),
            tool_google_search_aquarium()
        ]

        promptTemplate = PromptTemplate.from_template(
            template=f"""
            Du bist ein Aquarium-Experte und berätst einen Anfänger bei der Wahl 
            seines ersten Aquariums, der Technik, des Besatzes und der Bepflanzung. 
            Du gehst immer Schritt für Schritt vor und wählst erst ein Aquarium aus.
            Dann suchst du die passende Technik und schließlich den Besatz und die Bepflanzung.

            1. Suche ein passendes Aquarium aus der Aquarium-Datenbank in der Tabelle "Aquarium" aus.
               - Bedingungen:
                 - Kantenlänge: Weniger als oder gleich {request.availableSpace} cm
                 - Volumen: Weniger als oder gleich {request.maxVolume} Liter, aber nicht weniger als 54 Liter
                 - Preis: Weniger als oder gleich {request.maxCost}
               - Wenn du mehr als ein passendes Aquarium findest, wähle das größere aus.
               {f'- Suche dann bei Google, ob ein Unterschrank für dieses Aquarium existiert: {request.favoriteFishList}.' if request.needCabinet else ''}
               - **Stopping Condition:** Wenn ein passendes Aquarium gefunden wurde, gehe zu Schritt 2.

            2. Recherchiere dann die notwendige Technik für das gewählte Aquarium:
               - Verwende den Namen des Aquariums und suche bei Google nach geeigneter Technik.
               - Ist ein Filter, Beleuchtung und ein Heizer im Set enthalten?
               - Nur wenn nicht enthalten: Suche einen Filter, Beleuchtung und Heizer, die zum Aquarium passen.
               - Was sind die maximalen Kosten der Technik?
               - **Stopping Condition:** Wenn die empfohlene Technik und die maximalen Kosten gefunden wurden, gehe zu Schritt 3.

            3. Suche nun nach einem passenden Besatz und den entsprechenden Mengen:
               - Der Anfänger möchte {'einige' if request.favoritAnimals else 'keine'} bestimmte Tiere halten, sofern diese zum Aquarium und zu den Wasserwerten passen.
               {f'- Die folgenden Tiere sollten im Aquarium sein: {request.favoriteFishList}.' if request.favoritAnimals else ''}
               - Achte darauf, dass der Besatz auch untereinander verträglich ist.
               - Denke daran, dass die Person ein Anfänger ist.
               - Die Werte des Leitungswassers sind: {request.waterValues}.
               - **Stopping Condition:** Wenn ein passender Besatz gefunden wurde, gehe zu Schritt 4.

            4. Suche eine passende Bepflanzung aus:
               - Die Bepflanzung richtet sich nach den Bedürfnissen der von dir ausgesuchten Tiere und der zur Verfügung stehenden Technik, wie z.B. CO2-Versorgung.
               - Der Anfänger möchte {'unbedingt' if request.useForegroundPlants else 'keine'} Vordergrundpflanzen haben.
               - Das Aquarium soll {request.plantingIntensity} bepflanzt werden.
               - Bedenke bei der Auswahl der Pflanzen auch das Wachstum und die Pflege.
               - Der Pflegeaufwand soll maximal {request.maintenanceEffort} sein.
               - **Stopping Condition:** Wenn die optimale Bepflanzung gefunden wurde, fasse alle Ergebnisse zusammen und stelle sie in einer Markdown-Tabelle dar.
            """,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        prompt = hub.pull("hwchase17/react")
        react_agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True, early_stopping_method="generate", maxIterations=20)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
