import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_session():
    """Realistic session payload matching SessionInput."""
    return {
        "Administrative": 1,
        "Administrative_Duration": 14.65,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 19,
        "ProductRelated_Duration": 283.88,
        "BounceRates": 0.008,
        "ExitRates": 0.042,
        "PageValues": 68.58,
        "SpecialDay": 0.0,
        "Month": "June",
        "OperatingSystems": 1,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 1,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }
