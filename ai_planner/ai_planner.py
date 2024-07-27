import logging

from fastapi import FastAPI

from ai_planner_aquarium import planning_aquarium_controller
from ai_planner_animals import planning_animals_controller
from ai_planner_plants import planning_plants_controller
from model.input_model import RequestBody

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

# FastAPI config
app = FastAPI()


@app.post("/planner/")
async def chat(request: RequestBody):
    if request.planningMode == "Aquarium":
        return await planning_aquarium_controller(request)
    elif request.planningMode == "Besatz":
        return await planning_animals_controller(request)
    elif request.planningMode == "Pflanzen":
        return await planning_plants_controller(request)
