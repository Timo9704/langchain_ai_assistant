from dotenv import load_dotenv
from langchain import hub
import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chains.llm_math.base import LLMMathChain
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(
  page_title="Aquarium Assistant", layout="wide", initial_sidebar_state="collapsed"
)

def init_action():
  llm = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)
  llm_math_chain_tool = LLMMathChain.from_llm(llm)
  tools = [
    Tool(
      name="Calculator",
      func=llm_math_chain_tool.run,
      description="useful for when you need to answer questions about math"
    )
  ]
  prompt = hub.pull("hwchase17/react")
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
    answer = agent_executor.invoke({"input": user_input}, cfg)
    answer_container.write(answer["output"])


init_action()