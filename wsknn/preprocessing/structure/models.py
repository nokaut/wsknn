from typing import List, Union, Optional

from pydantic import BaseModel
from preprocessing.core.structure.properties.event_properties import EventPropertiesModel
from preprocessing.core.structure.properties.item_properties import ItemPropertiesModel


class SessionItemsMapModel(BaseModel):
    items: List[Union[int, str]] = []
    timestamps: List[Union[int, float]] = []
    purchase: List[bool] = []
    properties: Optional[List[EventPropertiesModel]] = None


class ItemSessionsMapModel(BaseModel):
    sessions: List[Union[int, str]] = []
    timestamps: List[Union[int, float]] = []
    properties: Optional[List[ItemPropertiesModel]] = None


class UserSessionsMapModel(BaseModel):
    user_id: str
    sessions: List[Union[int, str]]
