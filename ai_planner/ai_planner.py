import logging
from fastapi import FastAPI, HTTPException
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain_google_community import GoogleSearchAPIWrapper
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.llm_math.base import LLMMathChain
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

# FastAPI config
app = FastAPI()

llm = ChatOpenAI(model="gpt-4o", temperature=0.7, streaming=True)

def tool_retriever_vectorstore():
    embedding = OpenAIEmbeddings()
    pinecone_index = Pinecone.from_existing_index("aquabot", embedding=embedding)

    def retrieve_knowledge(query: str):
        results = pinecone_index.similarity_search(query, k=8)
        return results

    retriever_tool = Tool(
        name="Knowledge retriever",
        func=retrieve_knowledge,
        description="A knowledge base for aquaristic-topics. Useful for if you need to answer questions about aquarium, fish, aquatic plants or anything related to aquatics."
    )
    return retriever_tool


def tool_math_calculator():

    llm_math_chain_tool = LLMMathChain.from_llm(llm)

    calculator_tool = Tool(
        name="Calculator",
        func=llm_math_chain_tool.run,
        description="A math calculator. Useful for when you need to answer questions about math."
    )
    return calculator_tool


def tool_google_search():
    search = GoogleSearchAPIWrapper()

    google_search_tool = Tool(
        name="google_search",
        description="A web search. Useful for when you need to search for specific information"
                    " or if you need to verify answers from the knowledge retriever tool.",
        func=search.run,
    )
    return google_search_tool


@app.post("/planner/")
async def chat():
        try:
            tools = [
                tool_math_calculator(),
                tool_retriever_vectorstore(),
                tool_google_search()
            ]

            promptTemplate = PromptTemplate.from_template(
                template="""
            Du bist ein Aquarium-Experte und berätst einen Anfänger bei der Wahl seines ersten Aquariums, der Technik, des Besatzes und der Bepflanzung.
            Gehe Schritt für Schritt vor.
            1. Suche ein Aquarium, mit einer Kantenlänge von maximal 100cm. Der Anfänger wünscht sich ein Set-Aquarium. Das Set darf maximal 500 Euro kosten.
            2. Plane weitere Technik passend zu dem von dir ausgesuchten Aquarium. Wenn es ein Set-Aquarium mit Filter, Beleuchtung und Heizer ist, dannbeachte, dass die von dir ausgesuchte Technik sich nicht überschneidet.
            3. Suche Besatz, welcher zur Größe des Aquariums passt. Der Besatz muss auch untereinander vergesellschaftet werden. Denke daran, dass die Person ein Anfänger ist.
            4. Suche eine passende Bepflanzung aus. Die Bepflanzung richtet sich auch nach den Bedürfnissen der von dir ausgesuchten Tiere und der zur Verfügung stehenden Technik, wie z.B. CO2-Versorgung.
            5. Bereite deine Planung in einer Tabelle strukturiert auf. Hinterlege wenn möglich einen Link zu einem Online-Shop, bei dem das Produkt gekauft werden kann.
            """
            )

            prompt = hub.pull("hwchase17/react")
            print(prompt)
            react_agent = create_react_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)
            answer = agent_executor.invoke({"input": promptTemplate + " Bitte antworte in Deutsch!"})
            return {"answer": answer}
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

