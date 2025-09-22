from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models, schemas

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Ledger API", version="1.0.0")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/transactions/", response_model=schemas.TransactionRead)
def create_transaction(transaction: schemas.TransactionCreate, db: Session = Depends(get_db)):
    db_transaction = models.Transaction(
        type=transaction.type,
        created_at=transaction.created_at,
        customer_id=transaction.customer_id,
    )
    db.add(db_transaction)
    db.commit()
    db.refresh(db_transaction)

    for item in transaction.items:
        db_item = models.TransactionItem(
            quantity=item.quantity,
            unit_price=item.unit_price,
            total_price=item.quantity * item.unit_price,
            transaction_id=db_transaction.id,
            item_id=item.item_id,
        )
        db.add(db_item)

    db.commit()
    db.refresh(db_transaction)
    return db_transaction


@app.get("/transactions/{transaction_id}", response_model=schemas.TransactionRead)
def get_transaction(transaction_id: int, db: Session = Depends(get_db)):
    transaction = db.query(models.Transaction).filter(models.Transaction.id == transaction_id).first()
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")

    total_amount = sum(item.total_price for item in transaction.items)

    items = [
        {
            "id": item.id,
            "item_name": item.item.name if item.item else "Unknown",
            "quantity": item.quantity,
            "unit_price": item.unit_price,
            "total_price": item.total_price,
        }
        for item in transaction.items
    ]

    return {
        "id": transaction.id,
        "created_at": transaction.created_at,
        "type": transaction.type,
        "customer_id": transaction.customer_id,
        "customer_name": transaction.customer.name if transaction.customer else None,
        "items": items,
        "total_amount": total_amount,
    }


@app.get("/transactions/", response_model=list[schemas.TransactionRead])
def list_transactions(db: Session = Depends(get_db)):
    return db.query(models.Transaction).all()