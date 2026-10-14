"""Pydantic request and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PAYMENT_TYPES = (
    "ACH", "Cash Deposit", "Cash Withdrawal", "Cheque", "Credit card", "Cross-border", "Debit card",
)  # fmt: skip
CURRENCIES = (
    "Albanian lek", "Dirham", "Euro", "Indian rupee", "Mexican Peso", "Moroccan dirham", "Naira",
    "Pakistani rupee", "Swiss franc", "Turkish lira", "UK pounds", "US dollar", "Yen",
)  # fmt: skip
LOCATIONS = (
    "Albania", "Austria", "France", "Germany", "India", "Italy", "Japan", "Mexico", "Morocco",
    "Netherlands", "Nigeria", "Pakistan", "Spain", "Switzerland", "Turkey", "UAE", "UK", "USA",
)  # fmt: skip
MAX_AMOUNT = 1_000_000_000.0

PaymentType = Literal[*PAYMENT_TYPES]
Currency = Literal[*CURRENCIES]
Location = Literal[*LOCATIONS]


class TransactionRequest(BaseModel):
    """One transaction to score. Field names match the training dataset."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "Timestamp": "2023-08-01T12:30:00",
                "Sender_account": 8724731955,
                "Receiver_account": 2769355426,
                "Amount": 1459.15,
                "Payment_type": "Cash Deposit",
                "Payment_currency": "UK pounds",
                "Received_currency": "UK pounds",
                "Sender_bank_location": "UK",
                "Receiver_bank_location": "UK",
            }
        },
    )

    Timestamp: datetime = Field(description="Transaction time (timezone is ignored)")
    Sender_account: int = Field(ge=0, le=9_999_999_999)
    Receiver_account: int = Field(ge=0, le=9_999_999_999)
    Amount: float = Field(gt=0, le=MAX_AMOUNT, allow_inf_nan=False)
    Payment_type: PaymentType
    Payment_currency: Currency
    Received_currency: Currency
    Sender_bank_location: Location
    Receiver_bank_location: Location


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1] = Field(description="1 = flagged as suspicious")
    probability: float = Field(ge=0, le=1, description="Model score for the laundering class")
    threshold: float = Field(description="Decision threshold applied to the probability")
    model: str


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model: str | None = None
    detail: str | None = None
