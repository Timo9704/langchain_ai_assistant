from pydantic import BaseModel, Field


class Aquarium(BaseModel):
    identified_problems: str = Field(..., description="Erkannte Probleme speziell für das Aquarium und Technik, Beispiel: 1. XYZ 2. ABC")
    suggested_solutions: str = Field(..., description="detaillierte Lösungsvorschläge für die Probleme des Aquarium und Technik, Beispiel: 1. XYZ 2. ABC")

class Fish(BaseModel):
    identified_problems: str = Field(..., description="Erkannte Probleme speziell für die Fische, Beispiel: 1. XYZ 2. ABC")
    suggested_solutions: str = Field(..., description="detaillierte Lösungsvorschläge für die Probleme der Fische, Beispiel: 1. XYZ 2. ABC")

class Plant(BaseModel):
    identified_problems: str = Field(..., description="Erkannte Probleme speziell für die Pflanzen, Beispiel: 1. XYZ 2. ABC")
    suggested_solutions: str = Field(..., description="detaillierte Lösungsvorschläge für die Probleme die Pflanzen, Beispiel: 1. XYZ 2. ABC")


class AquariumOptimizerResult(BaseModel):
    aquarium: Aquarium
    fish: Fish
    plants: Plant
