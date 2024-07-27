from typing import List

from pydantic import BaseModel


class PlanningData(BaseModel):
    planningMode: str
    aquariumInfo: str
    availableSpace: int
    maxVolume: int
    needCabinet: bool
    maxCost: int
    favoritAnimals: bool
    favoriteFishList: str
    waterValues: str
    useForegroundPlants: bool
    plantingIntensity: str
    maintenanceEffort: str


class AquariumDataNoLink(BaseModel):
    aquariumName: str


class FishDataNoLink(BaseModel):
    fishName: str


class PlantDataNoLink(BaseModel):
    plantName: str


class PlanningDataNoLink(BaseModel):
    aquarium: AquariumDataNoLink
    fish: List[FishDataNoLink]
    plant: List[PlantDataNoLink]
