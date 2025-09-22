# app.py — ALISA Ledger (Production-Ready, Bilingual, Voice-Enabled, Reports)
# ----------------------------------------------------------------------------
# Run with:
#   streamlit run app.py
# ----------------------------------------------------------------------------

import os
import re
import unicodedata
from datetime import datetime, date, time as dtime
from io import BytesIO
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, joinedload, relationship

# Hybrid voice deps
from streamlit_mic_recorder import mic_recorder
import pyttsx3

# Optional deps
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

# ----------------------------------------------------------------------------
# Database setup
# ----------------------------------------------------------------------------

SQLALCHEMY_DATABASE_URL = "sqlite:///./ledger.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite concurrency
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)        # English name
    name_ur = Column(String, nullable=True)     # Urdu name
    phone = Column(String, nullable=True)
    address = Column(String, nullable=True)

    transactions = relationship("Transaction", back_populates="customer", cascade="all, delete-orphan")


class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)        # English name
    name_ur = Column(String, nullable=True)     # Urdu name
    price = Column(Float, default=0.0)

    transaction_items = relationship("TransactionItem", back_populates="item")


class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    type = Column(String, nullable=False)  # "debit" (sale) or "credit" (payment)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
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
    item_id = Column(Integer, ForeignKey("items.id"), nullable=True)  # nullable for payments

    transaction = relationship("Transaction", back_populates="items")
    item = relationship("Item", back_populates="transaction_items")


Base.metadata.create_all(bind=engine)


class DBSession:
    """Context manager for DB sessions"""
    def __enter__(self):
        self.db = SessionLocal()
        return self.db
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()


# ----------------------------------------------------------------------------
# Seed Defaults
# ----------------------------------------------------------------------------

def seed_default_data():
    with DBSession() as db:
        if db.query(Customer).count() == 0:
            db.add_all([
                Customer(name="Malik Ehsan",  name_ur="ملک احسان",  phone="03001234567", address="Jauharabad"),
                Customer(name="Malik Kamran", name_ur="ملک کامران", phone="03007654321", address="Khushab"),
            ])
            db.commit()
        if db.query(Item).count() == 0:
            db.add_all([
                Item(name="Khal Special", name_ur="کھل اسپیشل", price=2000),
                Item(name="Chokar",       name_ur="چوکر",      price=1500),
                Item(name="Oil Cake",     name_ur="تیل کھل",   price=2500),
            ])
            db.commit()                               
            # ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    """Normalize text for fuzzy Urdu/English matching."""
    if not s:
        return ""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8").lower().strip()


def tx_amount(tx: Transaction) -> float:
    """Compute transaction total amount."""
    return sum(item.total_price for item in tx.items)


def get_date_filter_range(option: str) -> Tuple[date, date]:
    """Return (start, end) date range based on filter option."""
    today = date.today()
    if option == "This Month":
        start = today.replace(day=1)
        end = today
    elif option == "Last Month":
        first = today.replace(day=1)
        end = first - pd.Timedelta(days=1)
        start = end.replace(day=1)
    else:
        start, end = None, None
    return start, end


def format_customer(c: Customer, lang: str) -> str:
    """Format customer name for UI."""
    if lang == "اردو":
        return c.name_ur or c.name or ""
    return c.name or c.name_ur or ""


def format_item(i: Item, lang: str) -> str:
    """Format item name for UI."""
    if lang == "اردو":
        return i.name_ur or i.name or ""
    return i.name or i.name_ur or ""


def get_openai_client(api_key: str):
    """Return OpenAI client if API key is provided."""
    if not api_key or not OpenAI:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None
    # ----------------------------------------------------------------------------
# PDF Invoice Generator
# ----------------------------------------------------------------------------
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


def generate_invoice_pdf(tx: Transaction, customer_name: str, balance: float, lang: str) -> bytes:
    """Generate a simple invoice PDF for a transaction."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, height - 2 * cm, "ALISA Ledger Invoice")
    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, height - 2.5 * cm, f"Customer: {customer_name}")
    c.drawString(2 * cm, height - 3 * cm, f"Transaction ID: {tx.id}")
    c.drawString(2 * cm, height - 3.5 * cm, f"Date: {tx.created_at.strftime('%Y-%m-%d')}")

    # Items table
    y = height - 5 * cm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2 * cm, y, "Item")
    c.drawString(8 * cm, y, "Qty")
    c.drawString(10 * cm, y, "Rate")
    c.drawString(13 * cm, y, "Total")
    c.setFont("Helvetica", 10)
    y -= 0.5 * cm

    for li in tx.items:
        name = li.item.name_ur if lang == "اردو" and li.item and li.item.name_ur else (li.item.name if li.item else "Payment")
        c.drawString(2 * cm, y, name)
        c.drawString(8 * cm, y, str(li.quantity))
        c.drawString(10 * cm, y, f"{li.unit_price:,.2f}")
        c.drawString(13 * cm, y, f"{li.total_price:,.2f}")
        y -= 0.5 * cm

    # Totals
    y -= 0.5 * cm
    total = tx_amount(tx)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(10 * cm, y, "Total:")
    c.drawString(13 * cm, y, f"{total:,.2f}")
    y -= 0.5 * cm
    c.drawString(10 * cm, y, "Balance After:")
    c.drawString(13 * cm, y, f"{balance:,.2f}")

    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


# ----------------------------------------------------------------------------
# Voice Handling
# ----------------------------------------------------------------------------

def speak_text(text: str, use_gtts: bool = False) -> None:
    """Convert text to speech either with pyttsx3 (offline) or gTTS (online)."""
    try:
        if use_gtts and gTTS:
            tts = gTTS(text=text, lang="ur")
            path = "temp_voice.mp3"
            tts.save(path)
            os.system(f"start {path}" if os.name == "nt" else f"mpg123 {path}")
        else:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        st.error(f"Voice error: {e}")


def transcribe_audio(client, audio_bytes: bytes) -> str:
    """Transcribe audio using OpenAI Whisper if available, else return empty."""
    if not client:
        return ""
    try:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        with open("temp_audio.wav", "rb") as f:
            resp = client.audio.transcriptions.create(model="whisper-1", file=f)
        return resp.text.strip()
    except Exception:
        return ""
    # ----------------------------------------------------------------------------
# Smart Agent (Fuzzy Match & Actions)
# ----------------------------------------------------------------------------

def smart_agent(command: str, lang: str, db) -> str:
    """Interpret a voice/text command and execute ledger actions."""
    norm_cmd = normalize_text(command)

    # Check if asking for balance of a customer
    if "balance" in norm_cmd or "khata" in norm_cmd or "کھاتہ" in command:
        custs = db.query(Customer).all()
        for c in custs:
            if normalize_text(c.name) in norm_cmd or normalize_text(c.name_ur) in norm_cmd:
                # Compute balance
                txs = db.query(Transaction).options(joinedload(Transaction.items)).filter(
                    Transaction.customer_id == c.id
                ).all()
                bal = 0.0
                for t in txs:
                    amt = tx_amount(t)
                    if t.type == "debit":
                        bal += amt
                    else:
                        bal -= amt
                return f"Balance for {c.name or c.name_ur}: {bal:,.2f}"
        return "Customer not found."

    # If not recognized
    return "Sorry, I did not understand the command."


# ----------------------------------------------------------------------------
# Customer Management UI
# ----------------------------------------------------------------------------

def manage_customers(lang: str):
    st.subheader("👥 Customers / گاہک")

    with DBSession() as db:
        custs = db.query(Customer).all()

    st.markdown("### ➕ Add Customer")
    name = st.text_input("Name (English)")
    name_ur = st.text_input("Name (Urdu)")
    phone = st.text_input("Phone")
    addr = st.text_input("Address")

    if st.button("Save Customer"):
        with DBSession() as db:
            db.add(Customer(name=name, name_ur=name_ur, phone=phone, address=addr))
            db.commit()
            st.success("Customer added!")
            st.experimental_rerun()

    st.markdown("---")
    st.markdown("### 📋 Existing Customers")

    for c in custs:
        st.write(f"**{format_customer(c, lang)}** — {c.phone or ''} — {c.address or ''}")
        if st.button(f"Delete {c.id}", key=f"del_cust_{c.id}"):
            with DBSession() as db:
                db.delete(db.query(Customer).get(c.id))
                db.commit()
                st.warning("Deleted!")
                st.experimental_rerun()


# ----------------------------------------------------------------------------
# Item Management UI
# ----------------------------------------------------------------------------

def manage_items(lang: str):
    st.subheader("📦 Items / اشیاء")

    with DBSession() as db:
        items = db.query(Item).all()

    st.markdown("### ➕ Add Item")
    name = st.text_input("Item Name (English)")
    name_ur = st.text_input("Item Name (Urdu)")
    price = st.number_input("Price", min_value=0.0, value=0.0)

    if st.button("Save Item"):
        with DBSession() as db:
            db.add(Item(name=name, name_ur=name_ur, price=price))
            db.commit()
            st.success("Item added!")
            st.experimental_rerun()

    st.markdown("---")
    st.markdown("### 📋 Existing Items")

    for it in items:
        st.write(f"**{format_item(it, lang)}** — {it.price:,.2f}")
        if st.button(f"Delete {it.id}", key=f"del_item_{it.id}"):
            with DBSession() as db:
                db.delete(db.query(Item).get(it.id))
                db.commit()
                st.warning("Deleted!")
                st.experimental_rerun()
                # ----------------------------------------------------------------------------
# Customer Ledger & Transactions
# ----------------------------------------------------------------------------

def customer_ledger_tab_content(lang: str):
    """Show ledger for a single customer, with running balance and add transactions."""
    st.subheader("📒 Customer Ledger / گاہک کا کھاتہ")

    with DBSession() as db:
        custs = db.query(Customer).order_by(Customer.id.asc()).all()

    sel_customer = st.selectbox(
        "Select Customer / گاہک منتخب کریں",
        custs,
        format_func=lambda c: format_customer(c, lang),
        key="ledger_customer"
    )

    date_opt = st.selectbox(
        "Date / تاریخ",
        ["All", "This Month", "Last Month", "Custom Range"],
        key="ledger_date_filter"
    )

    s, e = None, None
    if date_opt == "Custom Range":
        s = st.date_input("Start / شروع", key="ledger_date_start")
        e = st.date_input("End / اختتام", key="ledger_date_end")
        if s > e:
            st.error("❌ Invalid date range / تاریخ غلط")
            return
    elif date_opt in ["This Month", "Last Month"]:
        s, e = get_date_filter_range(date_opt)

    with DBSession() as db:
        q = db.query(Transaction).options(joinedload(Transaction.items).joinedload(TransactionItem.item)) \
            .filter(Transaction.customer_id == sel_customer.id)
        if s and e:
            q = q.filter(
                Transaction.created_at >= datetime.combine(s, dtime.min),
                Transaction.created_at <= datetime.combine(e, dtime.max)
            )
        txs = q.order_by(Transaction.created_at.asc()).all()

    # Running balance
    bal = 0.0
    rows = []
    for tx in txs:
        amt = tx_amount(tx)
        if tx.type == "debit":
            bal += amt   # Sale increases what they owe
        else:
            bal -= amt   # Payment reduces what they owe

        items_str = ", ".join(
            (li.item.name_ur if lang == "اردو" and li.item and li.item.name_ur
             else (li.item.name if li.item else ("ادائیگی" if lang == "اردو" else "Payment"))) + f"x{li.quantity}"
            for li in tx.items
        )
        rows.append({
            "Date": tx.created_at.strftime("%Y-%m-%d"),
            "Type": "فروخت" if (lang == "اردو" and tx.type == "debit") else ("Sale" if tx.type == "debit" else "Payment"),
            "Items": items_str,
            "Amount": amt,
            "Balance": bal
        })

    df = pd.DataFrame(rows, columns=["Date", "Type", "Items", "Amount", "Balance"])
    st.dataframe(df, use_container_width=True)

    # Export
    st.download_button(
        "⬇️ Export Ledger CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"ledger_{format_customer(sel_customer, lang)}.csv",
        mime="text/csv"
    )

    # Quick invoice for last tx
    if txs:
        last_tx = txs[-1]
        cname = format_customer(sel_customer, lang)
        pdf = generate_invoice_pdf(last_tx, cname, bal, lang)
        st.download_button(
            "⬇️ Download Last Tx Invoice PDF",
            data=pdf,
            file_name=f"invoice_{last_tx.id}.pdf",
            mime="application/pdf"
        )

    # ------------------------------------------------------------------------
    # Add Transaction (Sale or Payment)
    # ------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("➕ Add Transaction")

    with DBSession() as db:
        items_db = db.query(Item).order_by(Item.id.asc()).all()

    tx_type = st.selectbox("Type / قسم", ["debit (Sale)", "credit (Payment)"], key="ledger_tx_type")

    if "debit" in tx_type:
        n = st.number_input("Line items", min_value=1, value=1, step=1, key="ledger_sale_lines")
        sale_lines = []
        for i in range(int(n)):
            it = st.selectbox(f"Item {i+1}", items_db, format_func=lambda x: format_item(x, lang), key=f"ledger_sale_item_{i}")
            qty = st.number_input(f"Qty {i+1}", min_value=1, value=1, step=1, key=f"ledger_sale_qty_{i}")
            rate = st.number_input(f"Rate {i+1}", min_value=0.0, value=float(it.price or 0.0), key=f"ledger_sale_rate_{i}")
            sale_lines.append((it, int(qty), float(rate)))

        if st.button("💾 Save Sale / فروخت محفوظ کریں", key="ledger_save_sale"):
            with DBSession() as db:
                tx = Transaction(type="debit", customer_id=sel_customer.id, created_at=datetime.now())
                db.add(tx)
                for (it, q, r) in sale_lines:
                    db.add(TransactionItem(
                        transaction=tx,
                        item_id=it.id,
                        quantity=q,
                        unit_price=r,
                        total_price=q * r
                    ))
                db.commit()
                st.success("✅ Sale saved!")
                st.experimental_rerun()

    else:  # Payment
        amount = st.number_input("Payment Amount / ادائیگی", min_value=0.0, value=0.0, key="ledger_payment_amt")
        if st.button("💾 Save Payment / ادائیگی محفوظ کریں", key="ledger_save_payment"):
            with DBSession() as db:
                tx = Transaction(type="credit", customer_id=sel_customer.id, created_at=datetime.now())
                db.add(tx)
                db.add(TransactionItem(
                    transaction=tx,
                    item_id=None,
                    quantity=1,
                    unit_price=amount,
                    total_price=amount
                ))
                db.commit()
                st.success("✅ Payment saved!")
                st.experimental_rerun()
                # ----------------------------------------------------------------------------
# Shopkeeper Mode
# ----------------------------------------------------------------------------

def shopkeeper_mode(lang: str, client):
    st.subheader("🛍️ Shopkeeper Mode / دکاندار موڈ")

    # Daily summary
    with DBSession() as db:
        today = date.today()
        txs = db.query(Transaction).options(joinedload(Transaction.items)).filter(
            Transaction.created_at >= datetime.combine(today, dtime.min),
            Transaction.created_at <= datetime.combine(today, dtime.max)
        ).all()
        total_sales = sum(tx_amount(tx) for tx in txs if tx.type == "debit")
        total_payments = sum(tx_amount(tx) for tx in txs if tx.type == "credit")

        # Outstanding balance (positive = customers owe you)
        all_txs = db.query(Transaction).options(joinedload(Transaction.items)).all()
        balance = 0.0
        for t in all_txs:
            amt = tx_amount(t)
            if t.type == "debit":
                balance += amt
            else:
                balance -= amt

    col1, col2, col3 = st.columns(3)
    col1.metric("Today's Sales / آج کی فروخت", f"{total_sales:,.2f}")
    col2.metric("Today's Payments / آج کی ادائیگیاں", f"{total_payments:,.2f}")
    col3.metric("Outstanding Balance / بقیہ بیلنس", f"{balance:,.2f}")

    st.markdown("---")
    customer_ledger_tab_content(lang)


# ----------------------------------------------------------------------------
# Manager Mode
# ----------------------------------------------------------------------------

def manager_mode(lang: str, client):
    st.subheader("🛠️ Manager Mode / منیجر موڈ")

    tab_manage, tab_reports = st.tabs(["⚙ Manage / انتظام", "📊 Reports / رپورٹس"])

    # -----------------------
    # Manage Tab
    # -----------------------
    with tab_manage:
        st.markdown("### 👥 Manage Customers")
        manage_customers(lang)

        st.markdown("---")
        st.markdown("### 📦 Manage Items")
        manage_items(lang)

    # -----------------------
    # Reports Tab
    # -----------------------
    with tab_reports:
        st.markdown("### 📊 Customer Outstanding Balances")
        with DBSession() as db:
            custs = db.query(Customer).all()
            report_rows = []
            for c in custs:
                txs = db.query(Transaction).options(joinedload(Transaction.items)).filter(
                    Transaction.customer_id == c.id
                ).all()
                bal = 0.0
                for t in txs:
                    amt = tx_amount(t)
                    if t.type == "debit":
                        bal += amt
                    else:
                        bal -= amt
                report_rows.append({"Customer": format_customer(c, lang), "Balance": bal})

            df = pd.DataFrame(report_rows).sort_values("Balance", ascending=False)
            st.dataframe(df, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📈 Sales vs Payments Over Time")
        with DBSession() as db:
            txs = db.query(Transaction).all()
            if txs:
                df = pd.DataFrame([{
                    "Date": t.created_at.date(),
                    "Type": t.type,
                    "Amount": tx_amount(t)
                } for t in txs])
                daily = df.groupby(["Date", "Type"])["Amount"].sum().reset_index()
                pivot = daily.pivot(index="Date", columns="Type", values="Amount").fillna(0)
                st.line_chart(pivot)
            else:
                st.info("No transactions yet.")
                # ----------------------------------------------------------------------------
# Voice Assistant Tab
# ----------------------------------------------------------------------------

def voice_assistant_tab_content(lang: str, client):
    st.subheader("🎙️ Voice Assistant / آواز والا معاون")

    wav_audio = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="⏹ Stop Recording", key="voice_mic")

    if wav_audio is not None:
        # Handle dict vs bytes
        if isinstance(wav_audio, dict) and "bytes" in wav_audio:
            audio_bytes = wav_audio["bytes"]
        else:
            audio_bytes = wav_audio

        if audio_bytes:
            text = transcribe_audio(client, audio_bytes)
            if text:
                st.write(f"🗣️ You said: {text}")
                with DBSession() as db:
                    reply = smart_agent(text, lang, db)
                st.success(f"🤖 Assistant: {reply}")
                speak_text(reply, use_gtts=True)


# ----------------------------------------------------------------------------
# Main Entry Point
# ----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="ALISA Ledger", page_icon="📒", layout="wide")
    st.title("📒 ALISA Ledger — Bilingual AI Ledger")

    # Seed initial data if needed
    seed_default_data()

    # Sidebar settings
    lang = st.sidebar.radio("Language / زبان", ["English", "اردو"], key="lang_select")
    mode = st.sidebar.radio("Mode / موڈ", ["Shopkeeper", "Manager", "Voice Assistant"], key="mode_select")
    api_key = st.sidebar.text_input("🔑 OpenAI API Key (optional)", type="password")
    client = get_openai_client(api_key)

    # Route to mode
    if mode == "Shopkeeper":
        shopkeeper_mode(lang, client)
    elif mode == "Manager":
        manager_mode(lang, client)
    else:
        voice_assistant_tab_content(lang, client)


if __name__ == "__main__":
    main()
    