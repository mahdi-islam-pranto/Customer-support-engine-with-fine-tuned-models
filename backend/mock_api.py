"""
mock_api.py — Fake e-commerce API that simulates database operations.
In a real deployment, replace each function body with your actual DB/API calls.
"""

import random
import string
from datetime import datetime, timedelta
from typing import Any


# ── Fake database ────────────────────────────────────────────────────────────

FAKE_ORDERS = {
    "764283": {"status": "shipped",   "product": "Wireless Headphones",  "address": "10 Old St",        "eta": 2},
    "243565": {"status": "processing","product": "Running Shoes (UK 9)",  "address": "22 Baker Ave",     "eta": 4},
    "750175": {"status": "processing","product": "Coffee Maker",          "address": "5 Maple Rd",       "eta": 5},
    "789456": {"status": "processing","product": "Gaming Mouse",          "address": "88 Pine Blvd",     "eta": 3},
    "334455": {"status": "shipped",   "product": "Mechanical Keyboard",   "address": "14 Oak Lane",      "eta": 1},
    "998877": {"status": "delivered", "product": "USB-C Hub",             "address": "31 Elm Street",    "eta": 0},
    "112233": {"status": "processing","product": "Yoga Mat",              "address": "7 Birch Close",    "eta": 6},
}

FAKE_ACCOUNTS = {
    "jamie13@yahoo.com":   {"name": "Jamie Lee",    "plan": "platinum", "active": True},
    "john@example.com":    {"name": "John Smith",   "plan": "basic",    "active": True},
    "sarah@gmail.com":     {"name": "Sarah Connor", "plan": "premium",  "active": True},
}

CANCELLABLE_STATUSES   = {"processing", "pending"}
RETURNABLE_DAYS        = 30


def _random_ref() -> str:
    return "REF-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def _eta_string(days: int) -> str:
    if days == 0:
        return "already delivered"
    eta_date = datetime.now() + timedelta(days=days)
    return eta_date.strftime("%A, %d %B %Y")


# ── Action handlers ───────────────────────────────────────────────────────────

def cancel_order(order_id: str, **kwargs) -> dict[str, Any]:
    order = FAKE_ORDERS.get(order_id)
    if not order:
        return {"success": False, "message": f"Order #{order_id} not found."}
    if order["status"] not in CANCELLABLE_STATUSES:
        return {
            "success": False,
            "message": (
                f"Order #{order_id} cannot be cancelled — "
                f"it is already {order['status']}."
            ),
        }
    order["status"] = "cancelled"
    return {
        "success": True,
        "message": (
            f"Order #{order_id} ({order['product']}) has been cancelled. "
            f"A refund will be processed within 3–5 business days."
        ),
        "reference": _random_ref(),
    }


def change_order(order_id: str, **kwargs) -> dict[str, Any]:
    order = FAKE_ORDERS.get(order_id)
    if not order:
        return {"success": False, "message": f"Order #{order_id} not found."}
    if order["status"] not in CANCELLABLE_STATUSES:
        return {
            "success": False,
            "message": (
                f"Order #{order_id} is already {order['status']} "
                f"and can no longer be modified."
            ),
        }
    return {
        "success": True,
        "message": (
            f"Order #{order_id} is open for changes. "
            f"Please specify what you'd like to modify (size, colour, quantity)."
        ),
    }


def set_up_shipping_address(order_id: str, new_address: str, **kwargs) -> dict[str, Any]:
    order = FAKE_ORDERS.get(order_id)
    if not order:
        return {"success": False, "message": f"Order #{order_id} not found."}
    if order["status"] not in CANCELLABLE_STATUSES:
        return {
            "success": False,
            "message": (
                f"Order #{order_id} is already {order['status']}. "
                f"Address cannot be changed after dispatch."
            ),
        }
    old_address = order["address"]
    order["address"] = new_address
    return {
        "success": True,
        "message": (
            f"Delivery address for order #{order_id} updated.\n"
            f"  Old: {old_address}\n"
            f"  New: {new_address}"
        ),
        "reference": _random_ref(),
    }


def track_order(order_id: str, **kwargs) -> dict[str, Any]:
    order = FAKE_ORDERS.get(order_id)
    if not order:
        return {"success": False, "message": f"Order #{order_id} not found."}
    return {
        "success": True,
        "message": (
            f"Order #{order_id} — {order['product']}\n"
            f"  Status : {order['status'].capitalize()}\n"
            f"  Address: {order['address']}\n"
            f"  ETA    : {_eta_string(order['eta'])}"
        ),
    }


def return_item(order_id: str, **kwargs) -> dict[str, Any]:
    order = FAKE_ORDERS.get(order_id)
    if not order:
        return {"success": False, "message": f"Order #{order_id} not found."}
    if order["status"] != "delivered":
        return {
            "success": False,
            "message": f"Order #{order_id} has not been delivered yet — returns open after delivery.",
        }
    return {
        "success": True,
        "message": (
            f"Return initiated for order #{order_id} ({order['product']}). "
            f"A prepaid return label will be emailed to you within 24 hours. "
            f"Refund processed in 5–7 business days after we receive the item."
        ),
        "reference": _random_ref(),
    }


def delete_account(email: str, **kwargs) -> dict[str, Any]:
    account = FAKE_ACCOUNTS.get(email)
    if not account:
        return {"success": False, "message": f"No account found for {email}."}
    if not account["active"]:
        return {"success": False, "message": f"Account {email} is already deactivated."}
    account["active"] = False
    return {
        "success": True,
        "message": (
            f"Account for {account['name']} ({email}) has been scheduled for deletion. "
            f"All data will be permanently removed within 30 days per our privacy policy."
        ),
        "reference": _random_ref(),
    }


def recover_password(email: str, **kwargs) -> dict[str, Any]:
    account = FAKE_ACCOUNTS.get(email)
    if not account:
        return {
            "success": True,  # don't leak whether email exists
            "message": (
                f"If an account exists for {email}, "
                f"a password reset link has been sent."
            ),
        }
    return {
        "success": True,
        "message": (
            f"A password reset link has been sent to {email}. "
            f"It expires in 15 minutes."
        ),
    }


def get_invoice(order_id: str, **kwargs) -> dict[str, Any]:
    order = FAKE_ORDERS.get(order_id)
    if not order:
        return {"success": False, "message": f"Order #{order_id} not found."}
    return {
        "success": True,
        "message": (
            f"Invoice for order #{order_id} ({order['product']}) "
            f"has been sent to your registered email address."
        ),
        "reference": _random_ref(),
    }


# ── Dispatch table ────────────────────────────────────────────────────────────

INTENT_HANDLERS: dict[str, Any] = {
    "cancel_order":            cancel_order,
    "change_order":            change_order,
    "set_up_shipping_address": set_up_shipping_address,
    "track_order":             track_order,
    "return_item":             return_item,
    "delete_account":          delete_account,
    "recover_password":        recover_password,
    "get_invoice":             get_invoice,
}


def execute_action(intent: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    Main entry point called by main.py.
    Looks up the intent handler and calls it with the extracted params.
    """
    handler = INTENT_HANDLERS.get(intent)
    if handler is None:
        return {
            "success": False,
            "message": (
                f"I understood you want to perform '{intent}', "
                f"but I don't have a handler for that action yet."
            ),
        }
    try:
        return handler(**params)
    except TypeError as e:
        return {
            "success": False,
            "message": f"Missing required information to complete this action: {e}",
        }