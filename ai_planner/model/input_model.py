from typing import List

from pydantic import BaseModel


class PlanningData(BaseModel):
    planningMode: str
    aquariumInfo: str
    availableSpace: int
    minVolume: int
    maxVolume: int
    needCabinet: bool
    isSet: bool
    maxCost: int
    favoriteFishList: str
    waterValues: str
    useForegroundPlants: bool
    useMossPlants: bool
    growthRate: str


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
