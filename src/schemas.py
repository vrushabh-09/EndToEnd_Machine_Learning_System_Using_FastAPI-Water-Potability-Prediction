"""
Request/response schemas for the Water Potability API.

Fields are Optional[float] because the trained pipeline includes a median
imputer — a caller who doesn't have a Sulfate or Trihalomethanes reading
(common in the real world; ~24% and ~5% missing in the training data
respectively) can simply omit that field and still get a prediction.

Numeric bounds are generous physical/plausible limits (not just the dataset's
observed min/max) so obviously-wrong input (negative concentrations, pH of
40) is rejected with a clear 422 instead of silently mispredicting.
"""
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class WaterSample(BaseModel):
    ph: Optional[float] = Field(
        default=None, ge=0, le=14, description="pH of water (0-14 scale)"
    )
    Hardness: Optional[float] = Field(
        default=None, ge=0, le=1000, description="Water hardness in mg/L"
    )
    Solids: Optional[float] = Field(
        default=None, ge=0, le=100000, description="Total dissolved solids in ppm"
    )
    Chloramines: Optional[float] = Field(
        default=None, ge=0, le=50, description="Chloramines concentration in ppm"
    )
    Sulfate: Optional[float] = Field(
        default=None, ge=0, le=1000, description="Sulfate concentration in mg/L"
    )
    Conductivity: Optional[float] = Field(
        default=None, ge=0, le=2000, description="Electrical conductivity in \u03bcS/cm"
    )
    Organic_carbon: Optional[float] = Field(
        default=None, ge=0, le=100, description="Organic carbon in ppm"
    )
    Trihalomethanes: Optional[float] = Field(
        default=None, ge=0, le=250, description="Trihalomethanes in \u03bcg/L"
    )
    Turbidity: Optional[float] = Field(
        default=None, ge=0, le=20, description="Turbidity in NTU"
    )

    @model_validator(mode="after")
    def at_least_one_field(self):
        if all(v is None for v in self.model_dump().values()):
            raise ValueError("Provide at least one water quality measurement.")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "ph": 7.1,
                "Hardness": 196.9,
                "Solids": 20927.8,
                "Chloramines": 7.1,
                "Sulfate": 333.1,
                "Conductivity": 421.9,
                "Organic_carbon": 14.2,
                "Trihalomethanes": 66.6,
                "Turbidity": 3.9,
            }
        }
    }


class PredictionResponse(BaseModel):
    potable: bool
    label: str
    probability_potable: float = Field(..., description="Model probability that the sample is potable")
    confidence: str = Field(..., description="Human-readable confidence bucket")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
