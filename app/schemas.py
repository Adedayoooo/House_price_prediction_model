from pydantic import BaseModel, Field
from typing import Optional, Literal

class HouseFeatures(BaseModel):
    YearBuilt: int = Field(..., ge=1900, le=2025, description="Year the house was built")
    Floors: int = Field(..., ge=1, le=4, description="Number of floors")
    Bathrooms: int = Field(..., ge=0.5, le=8, description="Number of bathrooms")
    Area: float = Field(..., ge=300, description="Living area in square feet")
    Location: Literal["urban", "suburban", "rural"] = Field(...,description="Location type: urban, suburban, or rural")
    
class PredictionResponse(BaseModel):
    predicted_price: float
    formatted_price: str
    status: str = "success"

class ErrorResponse(BaseModel):
    """Error response"""
    detail: str
    error_code: Optional[str] = None
