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
    aquarium_name: str


class FishDataNoLink(BaseModel):
    fish_lat_name: str


class PlantDataNoLink(BaseModel):
    plant_name: str


class PlanningDataNoLink(BaseModel):
    aquarium: AquariumDataNoLink
    fishes: List[FishDataNoLink]
    plants: List[PlantDataNoLink]
