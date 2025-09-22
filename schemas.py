from pydantic import BaseModel
from typing import List, Optional


class TransactionItemCreate(BaseModel):
    item_id: int
    quantity: int
    unit_price: float


class TransactionItemRead(BaseModel):
    id: int
    item_name: str
    quantity: int
    unit_price: float
    total_price: float


class TransactionCreate(BaseModel):
    type: str
    created_at: str
    customer_id: int
    items: List[TransactionItemCreate]


class TransactionRead(BaseModel):
    id: int
    created_at: str
    type: str
    customer_id: int
    customer_name: Optional[str] = None
    items: List[TransactionItemRead]
    total_amount: float


class CustomerRead(BaseModel):
    id: int
    name: str
    phone: Optional[str]
    address: Optional[str]