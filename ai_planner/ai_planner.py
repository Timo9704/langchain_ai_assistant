import logging

from fastapi import FastAPI, HTTPException

from ai_planner_links import search_links_controller
from ai_planner_aquarium import planning_aquarium_controller
from ai_planner_animals import planning_animals_controller
from ai_planner_plants import planning_plants_controller
from model.input_model import PlanningDataNoLink, PlanningData

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

# FastAPI config
app = FastAPI()


@app.post("/planner/")
async def chat(request: PlanningData):
    try:
        if request.planningMode == "Aquarium":
            return await planning_aquarium_controller(request)
        elif request.planningMode == "Besatz":
            return planning_animals_controller(request)
        elif request.planningMode == "Pflanzen":
            return planning_plants_controller(request)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/links/")
async def chat(request: PlanningDataNoLink):
    try:
        return await search_links_controller(request)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))