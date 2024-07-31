from pydantic import BaseModel


class RequestBody(BaseModel):
    aquariumInfo: str
    waterClear: bool
    waterTurbidity: str
    aquariumProblemDescription: str
    fishHealthProblem: bool
    fishDiverseFeed: bool
    fishProblemDescription: str
    plantGrowthProblem: bool
    plantDeficiencySymptom: bool
    plantDeficiencySymptomDescription: str
    plantProblemDescription: str
