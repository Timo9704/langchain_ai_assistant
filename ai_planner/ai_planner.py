import logging
import os

from fastapi import FastAPI, HTTPException
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain_google_community import GoogleSearchAPIWrapper
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.llm_math.base import LLMMathChain
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

from model.input_model import RequestBody

load_dotenv()

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

# FastAPI config
app = FastAPI()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
response_schemas = [
    ResponseSchema(name="aquarium_name", description="Name des Aquariums oder des Aquarium-Sets"),
    ResponseSchema(name="aquarium_preis", description="Preis des Aquariums oder des Aquarium-Sets"),
    ResponseSchema(name="aquarium_liter", description="Liter des Aquariums oder des Aquarium-Sets"),
    ResponseSchema(name="aquarium_link", description="Link zum Online-Shop für das Aquarium oder das Aquarium-Set"),
    ResponseSchema(name="technik", description="Liste der enthaltenen Technik"),
    ResponseSchema(name="pflanzen", description="Array der enthaltenen Pflanzen")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

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
                    "Nutze diesen, wenn du keine ggeigneten Informationen in Search findest."
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
        name="Google Search Aquarium",
        description="Eine Websuche. Nützlich, wenn du nach Aquarien suchen möchtest. Hier kannst du Größen, "
                    "Literangaben oder Preise finden.",
        func=search.run
    )
    return google_search_tool


def tool_google_search_fish():
    search = GoogleSearchAPIWrapper(google_cse_id=os.environ.get("GOOGLE_CSE_ID_FISH"))

    google_search_tool = Tool(
        name="Google Search Fish",
        description="Eine Websuche. Nützlich, wenn du nach Fischen, Garnelen oder Schnecken suchen möchtest.",
        func=search.run
    )
    return google_search_tool


@app.post("/planner/")
async def chat(request: RequestBody):
        try:
            tools = [
                tool_math_calculator(),
                tool_retriever_vectorstore(),
                tool_google_search_aquarium()
            ]

            promptTemplate = PromptTemplate.from_template(
                template=f"""
            Du bist ein Aquarium-Experte und berätst einen Anfänger bei der Wahl seines ersten Aquariums, der Technik, des Besatzes und der Bepflanzung.
            Gehe Schritt für Schritt vor.
            1. Suche ein passendes Aquarium im Tool "Knowledge retriever". Weniger als {request.availableSpace}cm Kantenlänge und weniger als {request.maxVolume} Liter, aber mehr als 54 Liter. Der Anfänger hat ein Budget von maximal {request.maxCost} Euro und {'benötigt' if request.needCabinet else 'benötigt keinen'} zusätzlichen Unterschrank.
            2. Plane weitere Technik passend zu dem von dir ausgesuchten Aquarium. Überprüfe, ob es ein Set-Aquarium mit Filter, Beleuchtung und Heizer ist, falls ja, dann wird keine weitere Technik benötigt. Falls nein, dann suche nach Technik, welche zu deinem ausgesuchten Aquarium passt.
            3. Suche nun passenden Besatz. Der Anfänger hat {'einige' if request.favoritAnimals else 'keine'} Tiere, welche er unbedingt halten möchte, sofern diese zum Aquarium und zu den Wasserwerten passen. {f'Die folgenden Tiere sollten sind Aquarium: {request.favoriteFishList}.' if request.favoritAnimals else ''} 
            Bedenke, dass der Besatz muss auch untereinander vergesellschaftet werden können muss. Denke daran, dass die Person ein Anfänger ist. Die Werte des Leitungswassers sind: {request.waterValues}.
            4. Suche eine passende Bepflanzung aus. Die Bepflanzung richtet sich auch nach den Bedürfnissen der von dir ausgesuchten Tiere und der zur Verfügung stehenden Technik, wie z.B. CO2-Versorgung. Der Anfänger möchte {'unbedingt' if request.useForegroundPlants else 'keine'} Vordergrundpflanzen haben. Das Aquarium soll {request.plantingIntensity} bepflanzt werden.
            Bedenke bei der Auswahl der Pflanzen auch das Wachstum und die Pflege. Der Pflegeaufwand soll maximal {request.maintenanceEffort} sein.
            Bitte antworte in Deutsch!""",
                partial_variables={"format_instructions": format_instructions},
            )

            print(promptTemplate)

            prompt = hub.pull("hwchase17/react")
            print(prompt)
            react_agent = create_react_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True)
            answer = agent_executor.invoke({"input": promptTemplate})["output"]
            return {"answer": answer}
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))