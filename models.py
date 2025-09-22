# models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from database import Base


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)       # English name
    name_ur = Column(String, nullable=True)    # Urdu name
    phone = Column(String, nullable=True)
    address = Column(String, nullable=True)

    transactions = relationship("Transaction", back_populates="customer")


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)       # English name
    name_ur = Column(String, nullable=True)    # Urdu name
    price = Column(Float, default=0.0)

    transaction_items = relationship("TransactionItem", back_populates="item")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String, nullable=False)  # "debit" (sale) or "credit" (payment)
    created_at = Column(DateTime, default=datetime.now)
    customer_id = Column(Integer, ForeignKey("customers.id"))

    customer = relationship("Customer", back_populates="transactions")
    items = relationship("TransactionItem", back_populates="transaction", cascade="all, delete-orphan")


class TransactionItem(Base):
    __tablename__ = "transaction_items"

    id = Column(Integer, primary_key=True, index=True)
    quantity = Column(Integer, default=1)
    unit_price = Column(Float, default=0.0)
    total_price = Column(Float, default=0.0)

    transaction_id = Column(Integer, ForeignKey("transactions.id"))
    item_id = Column(Integer, ForeignKey("items.id"), nullable=True)  # null for payments

    transaction = relationship("Transaction", back_populates="items")
    item = relationship("Item", back_populates="transaction_items")