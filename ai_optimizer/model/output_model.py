from pydantic import BaseModel, Field
from typing import List


class Aquarium(BaseModel):
    aquarium_name: str = Field(..., description="Name des Aquarium-Produkts")
    aquarium_price: str = Field(..., description="Preis des Aquariums")
    link: str = Field(..., description="Link zum Online-Shop für das Aquarium")
    included_items: List[str] = Field(..., description="Liste der im Set enthaltenen Artikel")
    cabinet: str = Field(..., description="Aquarium besitzt einen oder keinen Unterschrank")
    cabinet_price: str = Field(..., description="Preis des Unterschranks")


class Tech(BaseModel):
    tech_name: str = Field(..., description="Name des Technikprodukts")
    price: str = Field(..., description="Preis des Technikprodukts")
    link: str = Field(..., description="Link zum Online-Shop für das Technikprodukt")


class Fish(BaseModel):
    fish_common_name: str = Field(..., description="Umgangsprachlicher Name des Fisches")
    quantity: int = Field(..., description="Anzahl der Fischart")
    link: str = Field(..., description="Link zum Online-Shop für den Fisch")


class Plant(BaseModel):
    plant_name: str = Field(..., description="Name der Pflanze")
    quantity: int = Field(..., description="Anzahl der Pflanzen")
    link: str = Field(..., description="Link zum Online-Shop für die Pflanze")


class AquariumPlanningResult(BaseModel):
    aquarium: Aquarium
    tech: List[Tech]
    fish: List[Fish]
    plants: List[Plant]
