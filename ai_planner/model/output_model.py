from pydantic import BaseModel, Field
from typing import List, Union


class Aquarium(BaseModel):
    aquarium_name: str = Field(..., description="Name des Aquarium-Produkts")
    aquarium_length: str = Field(..., description="Länge des Aquariums")
    aquarium_depth: str = Field(..., description="Tiefe des Aquariums")
    aquarium_height: str = Field(..., description="Höhe des Aquariums")
    aquarium_liter: str = Field(..., description="Volumen des Aquariums")
    aquarium_price: str = Field(..., description="Preis des Aquariums")


class Filter(BaseModel):
    filter_name: str = Field(..., description="Modell-Name des Filters")
    filter_included: str = Field(..., description="Ist der Filter im Set enthalten?")


class Heater(BaseModel):
    heater_name: str = Field(..., description="Modell-Name des Heizers")
    heater_included: str = Field(..., description="Ist der Heizer im Set enthalten?")


class Lighting(BaseModel):
    lighting_name: str = Field(..., description="Modell-Name der Beleuchtung")
    lighting_included: str = Field(..., description="Ist die Beleuchtung im Set enthalten?")


class Fish(BaseModel):
    fish_common_name: str = Field(..., description="Umgangsprachlicher Name des Fisches")
    fish_lat_name: str = Field(..., description="Lateinischer Name des Fisches")
    fish_ph: str = Field(..., description="pH-Wert-Bereich des Wassers, in dem der Fisch gehalten werden kann")
    fish_gh: str = Field(..., description="GH-Wert-Bereich des Wassers, in dem der Fisch gehalten werden kann")
    fish_kh: str = Field(..., description="KH-Wert-Bereich des Wassers, in dem der Fisch gehalten werden kann")
    fish_min_temp: str = Field(..., description="Temperatur-Bereich, bei der der Fisch gehalten werden kann")
    fish_min_liters: str = Field(..., description="Ab wie viel Litern der Fisch gehalten werden kann")


class Plant(BaseModel):
    plant_name: str = Field(..., description="Name der Pflanze")
    plant_type: str = Field(..., description="Typ der Pflanze, z.B. Vordergrund, Mittelgrund, Hintergrund")
    plant_growth_rate: str = Field(..., description="Wachstumsgeschwindigkeit der Pflanze")
    plant_light_demand: str = Field(..., description="Lichtbedarf der Pflanze")
    plant_co2_demand: str = Field(..., description="CO2-Bedarf der Pflanze")


class FishesPlanningResult(BaseModel):
    fishes: List[Fish]
    reason: str = Field(..., description="Begründung für die Auswahl der Fische")


class PlantsPlanningResult(BaseModel):
    plants: List[Plant]
    reason: str = Field(..., description="Begründung für die Auswahl der Pflanzen")


class AquariumPlanningResult(BaseModel):
    aquarium: Aquarium
    technic: List[Union[Filter, Heater, Lighting]]
    fishes: List[Fish]
    plants: List[Plant]
    reason: str = Field(..., description="Begründung für die Auswahl des Aquariums,der Fische und Pflanzen")


class AquariumLink(BaseModel):
    aquarium_name: str = Field(..., description="Name des Aquarium-Produkts")
    link: str = Field(..., description="Link zum Online-Shop für das Aquarium")


class FishLink(BaseModel):
    fish_lat_name: str = Field(..., description="Lateinischer Name des Fisches")
    fish_link: str = Field(..., description="Link zum Online-Shop für den Fisch")


class PlantLink(BaseModel):
    plant_name: str = Field(..., description="Name der Pflanze")
    plant_link: str = Field(..., description="Link zum Online-Shop für die Pflanze")


class PlanningDataLink(BaseModel):
    aquarium: AquariumLink
    fishes: List[FishLink]
    plants: List[PlantLink]
