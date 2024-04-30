from dotenv import load_dotenv
from langchain.chains import LLMCheckerChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv()


def checker():
    llm = OpenAI(temperature=0.7)

    text = "Ist Nitrat giftig für mein Aquarium?"

    checker_chain = LLMCheckerChain.from_llm(llm)

    response = checker_chain.invoke(text)

    print(response)


def pdfsummary():
    pdf_file_path = ("C:\\Users\\timos\\OneDrive\\Studium\\Master\\Masterarbeit\\Masterarbeit\\Quellen\\KMI15_Modulhandbuch_HfTL.pdf")
    pdf_loader = PyPDFLoader(pdf_file_path)
    docs = pdf_loader.load_and_split()
    llm = OpenAI()
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    chain.invoke(docs)

def getToken():
    llm_chain = PromptTemplate.from_template("Tell me the list:") | OpenAI()

    input_list = [
        {"product": "socks"}
    ]
    llm_chain


getToken()
