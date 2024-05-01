from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone

load_dotenv()

loader = DirectoryLoader(
    "./algen/aquasabi",
    loader_cls=TextLoader
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# Embed
embedding = OpenAIEmbeddings()
vectorstore = Pinecone.from_existing_index("aquabot", embedding=embedding)
vectorstore.add_documents(splits)