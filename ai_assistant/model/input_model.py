from pydantic import BaseModel, Field
import uuid


def generate_session_id():
    return str(uuid.uuid4())


class WaterParameters(BaseModel):
    ph: float
    gh: int
    kh: int
    no2: float
    no3: int
    po4: float
    fe: float
    k: int


class AquariumData(BaseModel):
    aquarium_liter: int
    water_parameters: WaterParameters


class Preferences(BaseModel):
    experience_level: str
    detail_level: str


class AIInput(BaseModel):
    human_input: str
    session_id: str = Field(default_factory=generate_session_id)


class RequestBody(BaseModel):
    preferences: Preferences
    aquarium_data: AquariumData
    ai_input: AIInput
