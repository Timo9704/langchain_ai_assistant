from pydantic import BaseModel


class RequestBody(BaseModel):
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
