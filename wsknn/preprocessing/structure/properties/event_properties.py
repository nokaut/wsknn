from typing import Optional
from pydantic import BaseModel


class EventPropertiesModel(BaseModel):
    purchase: bool
    product_view: bool
    add_to_cart: Optional[bool] = None
    add_to_wishlist: Optional[bool] = None
