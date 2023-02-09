from typing import Union, Optional
from pydantic import BaseModel


class ItemPropertiesModel(BaseModel):
    price: Optional[float] = None
    category: Optional[Union[str, int]] = None
