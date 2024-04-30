from dotenv import load_dotenv
from langchain.chains import LLMCheckerChain
from langchain_openai import OpenAI

load_dotenv()

llm = OpenAI(temperature=0.7)

text = "Ist Nitrat giftig für mein Aquarium?"

checker_chain = LLMCheckerChain.from_llm(llm)

response = checker_chain.invoke(text)

print(response)
