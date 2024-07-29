from multiprocessing import Pool
from fastapi import FastAPI
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_community import GoogleSearchAPIWrapper

from model.output_model import PlanningDataLink
from model.input_model import PlanningDataNoLink, AquariumDataNoLink, FishDataNoLink, PlantDataNoLink

load_dotenv()

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

# FastAPI config
app = FastAPI()

# LLM config
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


async def search_links_controller(request: PlanningDataNoLink):
    items_to_process = [request.aquarium] + request.fishes + request.plants

    pool = Pool()
    results = [pool.apply_async(search_links, [item]) for item in items_to_process]

    answers = [result.get() for result in results]
    pool.close()
    pool.join()

    structured_answer = convert_to_json(*answers)
    return structured_answer


def convert_to_json(*answers):
    structured_llm = llm.with_structured_output(PlanningDataLink)
    structured_answer = structured_llm.invoke(" ".join(answers))
    return structured_answer


def search_links(item):
    search_wrappers = {
        AquariumDataNoLink: GoogleSearchAPIWrapper(google_cse_id=os.environ.get("GOOGLE_CSE_ID_AQUARIUM")),
        FishDataNoLink: GoogleSearchAPIWrapper(google_cse_id=os.environ.get("GOOGLE_CSE_ID_FISHES")),
        PlantDataNoLink: GoogleSearchAPIWrapper(google_cse_id=os.environ.get("GOOGLE_CSE_ID_PLANTS"))
    }

    search_function = search_wrappers[type(item)].results

    name = getattr(item, 'fish_lat_name', None) or getattr(item, 'plant_name', None) or getattr(item, 'aquarium_name',
                                                                                                "")
    if name:
        results = search_function(name, 2)
        if results:
            link = results[0]['link']
            if 'aquasabi' in link:
                link += '?ref=ak'  # Add referral parameter if the link is from Aquasabi
            output = f"Name: {name}, Link: {link}"
        else:
            output = f"Name: {name}, Link: kein Link gefunden"
    else:
        output = ""
    return output
