import os
from time import sleep

from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from scraper_aquasabi import extract_aquasabi_content
from sraper_garnelenhaus import extract_garnelenhaus_content
from webscraper.scraper_aquaristikprofi import extract_aquaristikprofi_content
from webscraper.sraper_garnelenguemmer import extract_garnelenguemmer_content


def read_urls_from_directory(directory_path):
    urls = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                urls.extend(file.read().splitlines())
    return urls


def split_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4", chunk_size=400, chunk_overlap=10
    )
    splits = text_splitter.split_text(text)
    print("Splitted up to: ", len(splits))
    return splits

def upload_to_vectordatabase(splittedfiles):
    # Embed
    embedding = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index("aquabot", embedding=embedding)
    vectorstore.add_texts(splittedfiles)


def main(file_path):
    urls = read_urls_from_directory(file_path)
    splittedcontent = []

    load_dotenv()

    for url in urls:
        sleep(10)

        if "aquasabi" in url:
            splittedcontent = split_to_chunks(extract_aquasabi_content(url))
        if "garnelenhaus" in url:
            splittedcontent = split_to_chunks(extract_garnelenhaus_content(url))
        if "aquaristik-profi" in url:
            splittedcontent = split_to_chunks(extract_aquaristikprofi_content(url))
        if "garnelen-guemmer" in url:
            splittedcontent = split_to_chunks(extract_garnelenguemmer_content(url))

        upload_to_vectordatabase(splittedcontent)
        print("Successfully uploaded content of " + url)


if __name__ == '__main__':
    # Pfad zur Datei mit den URLs
    file_path = './sources/'
    main(file_path)
