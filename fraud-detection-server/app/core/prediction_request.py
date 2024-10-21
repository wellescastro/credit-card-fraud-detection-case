from datetime import datetime

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    trans_date_trans_time: datetime
    merchant: str
    category: str
    amt: float
    city: str
    state: str
    lat: float
    long: float
    city_pop: int
    job: str
    dob: datetime
    trans_num: str
    merch_lat: float
    merch_long: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "trans_date_trans_time": "2019-04-01T15:06:00",
                    "merchant": "Predovic Inc",
                    "category": "shopping_net",
                    "amt": 966.11,
                    "city": "Wales",
                    "state": "AK",
                    "lat": 64.7556,
                    "long": -165.6723,
                    "city_pop": 145,
                    "job": "Administrator, education",
                    "dob": "1939-11-09",
                    "trans_num": "a3806e984cec6ac0096d8184c64ad3a1",
                    "merch_lat": 65.468863,
                    "merch_long": -165.473127,
                }
            ]
        }
    }
