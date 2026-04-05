"""
Input validation for the Paris 15e property valuation API.

Validates query parameters for GET /score before they reach the model.
"""

from pydantic import BaseModel, Field, field_validator

ALLOWED_PROPERTY_TYPES = {"Appartement", "Maison"}

class ScoreRequest(BaseModel):
    """Checks that the five query parameters the user sends are valid:
    - surface is a positive number ≤ 1000
    - nb_room is a non-negative integer
    - property_type is exactly "Appartement" or "Maison"
    - section_id matches the correct format (8 digits + 2 uppercase letters)
    - price is a positive number
    """

    surface: float = Field(
        ...,
        gt=0,
        le=1_000,
        description="Living area in m² (must be > 0 and ≤ 1000)",
    )
    nb_room: int = Field(
        ...,
        ge=0,
        le=50,
        description="Number of main rooms (must be ≥ 0)",
    )
    property_type: str = Field(
        ...,
        description="Property type: 'Appartement' or 'Maison'",
    )
    section_id: str = Field(
        ...,
        min_length=10,
        max_length=12,
        description="Section code, e.g. '75115000CG'",
    )
    price: float = Field(
        ...,
        gt=0,
        description="Listed price in € (must be > 0)",
    )

    @field_validator("property_type")
    @classmethod
    def property_type_must_be_allowed(cls, v: str) -> str:
        if v not in ALLOWED_PROPERTY_TYPES:
            raise ValueError(
                f"property_type must be one of {sorted(ALLOWED_PROPERTY_TYPES)}, got '{v}'"
            )
        return v

    @field_validator("section_id")
    @classmethod
    def section_id_format(cls, v: str) -> str:
        stripped = v.strip()
        if not (stripped[:8].isdigit() and stripped[8:].isalpha() and stripped[8:].isupper()):
            raise ValueError(
                f"section_id must be 8 digits followed by 2 uppercase letters "
                f"(e.g. '75115000CG'), got '{v}'"
            )
        return stripped

    @property
    def is_appartement(self) -> int:
        return 1 if self.property_type == "Appartement" else 0

    # Reconstruct a synthetic parcel_id so score.py's predict() can derive section_id
    @property
    def parcel_id(self) -> str:
        return self.section_id + "0000"