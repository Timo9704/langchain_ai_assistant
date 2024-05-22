
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone

load_dotenv()

text_loader_kwargs={'encoding': 'utf-8'}
loader = DirectoryLoader(
    "./algen/",
    loader_cls=TextLoader,
    loader_kwargs=text_loader_kwargs
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# Embed
embedding = OpenAIEmbeddings()
vectorstore = Pinecone.from_existing_index("aquabot", embedding=embedding)
vectorstore.add_documents(splits)