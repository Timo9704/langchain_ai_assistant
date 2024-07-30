from pydantic import BaseModel, Field
from typing import List, Union, Optional


class Aquarium(BaseModel):
    aquarium_name: str = Field(..., description="Name des Aquarium-Produkts")
    aquarium_length: str = Field(..., description="Länge des Aquariums, Beispiel: 100 cm")
    aquarium_depth: str = Field(..., description="Tiefe des Aquariums, Beispiel: 40 cm")
    aquarium_height: str = Field(..., description="Höhe des Aquariums, Beispiel: 50 cm")
    aquarium_liter: str = Field(..., description="Volumen des Aquariums, Beispiel: 200 Liter")
    aquarium_price: str = Field(..., description="Preis des Aquariums, Beispiel: 200 Euro")
    aquarium_set: str = Field(..., description="Ist das Aquarium ein Set? Ja oder Nein?, Beispiel: Ja")
    aquarium_cabinet: str = Field(..., description="Unterschrank enthalten? Ja oder Nein?, Beispiel: Ja")


class Filter(BaseModel):
    filter_name: str = Field(..., description="Modell-Name des Filters")
    filter_included: str = Field(..., description="Ist der Filter im Set enthalten? Ja oder Nein?, Beispiel: Ja")


class Heater(BaseModel):
    heater_name: str = Field(..., description="Modell-Name des Heizers")
    heater_included: str = Field(..., description="Ist der Heizer im Set enthalten? Ja oder Nein?, Beispiel: Ja")


class Lighting(BaseModel):
    lighting_name: str = Field(..., description="Modell-Name der Beleuchtung")
    lighting_included: str = Field(..., description="Ist die Beleuchtung im Set enthalten? Ja oder Nein?, Beispiel: Ja")


class Fish(BaseModel):
    fish_common_name: str = Field(..., description="Umgangsprachlicher Name des Fisches")
    fish_lat_name: str = Field(..., description="Lateinischer Name des Fisches")
    fish_ph: str = Field(...,
                         description="pH-Wert-Bereich des Wassers, in dem der Fisch gehalten werden kann, Beispiel: 6-8")
    fish_gh: str = Field(...,
                         description="GH-Wert-Bereich des Wassers, in dem der Fisch gehalten werden kann,Beispiel: 5-15 °dGH")
    fish_kh: str = Field(...,
                         description="KH-Wert-Bereich des Wassers, in dem der Fisch gehalten werden kann: Beispiel: 0-0 °dKH")
    fish_min_temp: str = Field(...,
                               description="Temperatur-Bereich, bei der der Fisch gehalten werden kann, Beispiel: 24-28 °C")
    fish_min_liters: str = Field(...,
                                 description="Ab wie viel Litern der Fisch gehalten werden kann, Beispiel: ab 100 Liter")


class Plant(BaseModel):
    plant_name: str = Field(..., description="Name der Pflanze")
    plant_type: str = Field(..., description="Typ der Pflanze, z.B. Vordergrund, Mittelgrund, Hintergrund")
    plant_growth_rate: str = Field(..., description="Wachstumsgeschwindigkeit der Pflanze, Beispiel: langsam")
    plant_light_demand: str = Field(..., description="Lichtbedarf der Pflanze, Beispiel: mittel")
    plant_co2_demand: str = Field(..., description="CO2-Bedarf der Pflanze, Beispiel: hoch")


class FishesPlanningResult(BaseModel):
    fishes: List[Fish]
    reason: str = Field(..., description="Begründung für die Auswahl der Fische")


class PlantsPlanningResult(BaseModel):
    plant: List[Plant]
    foreground_plants: str = Field(..., description="Anzahl der Vordergrundpflanzen, Beispiel: 3 Stück")
    midground_plants: str = Field(..., description="Anzahl der Mittelgrundpflanzen, Beispiel: 3 Stück")
    background_plants: str = Field(..., description="Anzahl der Hintergrundpflanzen, Beispiel: 3 Stück")


class AquariumPlanningResult(BaseModel):
    aquarium: Aquarium
    technic: List[Union[Filter, Heater, Lighting]]
    fishes: List[Fish]
    plants: PlantsPlanningResult
    reason: str = Field(...,
                        description="Begründung für die Auswahl des Aquariums,der Fische und Pflanzen mit 3-4 Sätzen")


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
