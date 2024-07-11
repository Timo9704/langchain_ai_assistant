from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv
from langchain import hub
import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
from langchain.chains.llm_math.base import LLMMathChain
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool

load_dotenv()

st.set_page_config(
    page_title="Aquarium Assistant", layout="wide", initial_sidebar_state="collapsed"
)

llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

def tool_retriever_vectorstore():
    embedding = OpenAIEmbeddings()
    pinecone_index = Pinecone.from_existing_index("aquabot", embedding=embedding)

    def retrieve_knowledge(query: str):
        results = pinecone_index.similarity_search(query, k=8)
        return results

    retriever_tool = Tool(
        name="Knowledge retriever",
        func=retrieve_knowledge,
        description="useful for if you need to answer questions about aquarium, fish, aquatic plants or anything related to aquatics"
    )
    return retriever_tool


def tool_math_calculator():

    llm_math_chain_tool = LLMMathChain.from_llm(llm)

    calculator_tool = Tool(
        name="Calculator",
        func=llm_math_chain_tool.run,
        description="useful for when you need to answer questions about math"
    )
    return calculator_tool


def tool_google_search():
    search = GoogleSearchAPIWrapper()

    google_search_tool = Tool(
        name="google_search",
        description="Search Google for recent results.",
        func=search.run,
    )
    return google_search_tool


def init_action():
    tools = [
        tool_math_calculator(),
        #tool_retriever_vectorstore(),
        tool_google_search()
    ]

    prompt = hub.pull("hwchase17/react")
    print(prompt)
    react_agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)

    with st.form(key="form"):
        user_input = st.text_input("Stelle deine Frage:")
        submit_clicked = st.form_submit_button("Stelle Frage")

    output_container = st.empty()
    if submit_clicked:
        output_container = output_container.container()
        output_container.chat_message("user").write(user_input)

        answer_container = output_container.chat_message("assistant")
        st_callback = StreamlitCallbackHandler(answer_container)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_callback]
        answer = agent_executor.invoke({"input": user_input + " Bitte antworte in Deutsch!"}, cfg)
        answer_container.write(answer["output"])


init_action()
