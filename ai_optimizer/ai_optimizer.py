import logging
import time

from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from ai_optimizer_fish import optimize_fish
from model.output_model import AquariumOptimizerResult, Aquarium
from ai_optimizer_aquarium import optimize_aquarium
from ai_optimizer_plants import optimize_plants
from model.input_model import RequestBody
from concurrent.futures import ProcessPoolExecutor, as_completed

load_dotenv()

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

# FastAPI config
app = FastAPI()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def convert_to_json(answer1, answer2, answer3):
    structured_llm = llm.with_structured_output(Aquarium)
    prompt = '''Hier ist das Ergebnis der Aquariumsoptimierung. Führe die Ergebnisse zusammen und erläutere die Lösungsvorschläge ausführlich. '''
    structured_answer = structured_llm.invoke(prompt + str(answer1) + " " + str(answer2) + " " + str(answer3))
    return structured_answer


@app.post("/optimizer/")
async def planning_aquarium_controller(request: RequestBody):
    start_time = time.time()
    answers = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(optimize_aquarium, request),
            executor.submit(optimize_plants, request),
            executor.submit(optimize_fish, request)
        ]

        for future in as_completed(futures):
            answers.append(future.result())

    structured_answer = convert_to_json(*answers)
    end_time = time.time()
    logger.info(f"Planning Aquarium took {end_time - start_time} seconds")
    return structured_answer
