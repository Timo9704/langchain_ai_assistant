import logging
import os
import time

from fastapi import HTTPException
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

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def tool_retriever_vectorstore_general():
    embedding = OpenAIEmbeddings()
    pinecone_index = Pinecone.from_existing_index("aquabot", embedding=embedding)

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


def tool_retriever_vectorstore_plants():
    embedding = OpenAIEmbeddings()
    pinecone_index = Pinecone.from_existing_index("aiplannerplants", embedding=embedding)

    def retrieve_knowledge(query: str):
        results = pinecone_index.similarity_search(query, k=8)
        return results

    retriever_tool = Tool(
        name="Wissensdatenbank für Pflanzen",
        func=retrieve_knowledge,
        description="Eine Wissensdatenbank, die Informationen zu einer spezifischen Pflanze, wie Namen, Lichtbedarf oder CO2-Bedarf liefert."
    )
    return retriever_tool


def optimize_plants(request: RequestBody):
    time_start = time.time()
    try:
        tools = [
            tool_retriever_vectorstore_general(),
            tool_retriever_vectorstore_plants(),
        ]

        promptTemplate = PromptTemplate.from_template(
            template=f"""
            Du bist ein Aquarium-Experte und analysierst für Aquarianer deren Aquarien, um ihnen zu helfen, optimale Bedingungen zu schaffen. 
            Gehe Schritt für Schritt vor und nutze alle verfügbaren Informationen, um eine umfassende Beratung zu bieten.

            Aquarium-Details: {request.aquariumInfo} {request.aquariumTechInfo} {request.latest3Measurements}
            Problembeschreibung des Aquarianers: 
            {request.plantProblemDescription}
            {'Die Pflanzen im Aquarium wachsen nicht gut.' if request.plantGrowthProblem else 'Die Pflanzen wachsen gut.'}
            {'Die Pflanzen zeigen keinen Mangel an!.' if request.plantDeficiencySymptom else 'Die Pflanzen zeigen die folgenden Mangelerscheinungen:' + request.plantDeficiencySymptomDescription}
            
            1. Ermittle Probleme in Bezug auf die Pflanzen:
            - Sind der pH-Wert, GH-Wert, KH-Wert, CO2-Gehalt, Licht oder die Nährstoffe problematisch?
            - Worauf könnten die Mangelsymptome hindeuten?
            Beachte, dass nicht alle Informationen relevant sein müssen.
            
            2. Erarbeite Lösungsvorschläge für jedes erkannte Problem.

            Falls kein Problem vorliegt, sollte klar "Keine Probleme gefunden!" stehen.
            """,
        )

        prompt = hub.pull("hwchase17/react")
        react_agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True)
        answer = agent_executor.invoke({"input": promptTemplate})["output"]
        time_end = time.time()
        logger.info(f"Planning optimize_plants finished in {time_end - time_start} seconds")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
