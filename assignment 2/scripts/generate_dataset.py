from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


SEED = 42
STORE_NAME = "ShopSphere"
EXAMPLES_PER_CATEGORY = 40
TRAIN_PER_CATEGORY = 32
VAL_PER_CATEGORY = 4
TEST_PER_CATEGORY = 4

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "ecommerce_support"

INTRO_PHRASES = [
    "Hi,",
    "Hello,",
    "Hey support,",
    "Good morning,",
    "Good evening,",
    "",
]
ENDINGS = [
    "Please help.",
    "What should I do?",
    "Please advise.",
    "Can you check this for me?",
    "I need support with this.",
    "Kindly guide me.",
]
CUSTOMER_TYPES = [
    "I am a first-time customer.",
    "I order from your app regularly.",
    "This is my first order on your platform.",
    "I use your website often.",
    "",
]
PRODUCTS = [
    "wireless earbuds",
    "gaming mouse",
    "Bluetooth speaker",
    "running shoes",
    "smartwatch",
    "phone case",
    "laptop backpack",
    "electric kettle",
    "yoga mat",
    "USB-C charger",
    "desk lamp",
    "mechanical keyboard",
    "air fryer",
    "skin care kit",
    "power bank",
]
SIZES = ["XS", "S", "M", "L", "XL", "UK 7", "UK 8", "UK 9", "128GB", "256GB"]
COLORS = ["black", "blue", "green", "pink", "white", "gray", "navy"]
PAYMENT_METHODS = ["UPI", "credit card", "debit card", "net banking", "wallet", "COD"]
COUPONS = ["SAVE10", "WELCOME15", "FESTIVE20", "APPONLY5", "STYLE25", "NEWUSER50"]
ISSUES = [
    "the app keeps showing an error",
    "the tracking page has not updated",
    "I cannot see any useful status on the order page",
    "the website is giving me confusing messages",
]
TIME_PHRASES = [
    "yesterday",
    "two days ago",
    "three days ago",
    "last night",
    "this morning",
]
DELIVERY_WINDOWS = [
    "the promised delivery date was yesterday",
    "it was supposed to arrive today but there is still no delivery partner update",
    "the estimated delivery window has already passed",
    "the order is now later than the original delivery date",
]
RETURN_REASONS = [
    "I changed my mind",
    "the product does not match my expectation",
    "I no longer need it",
    "I ordered the wrong variant by mistake",
]
LOGIN_ISSUES = [
    "I am not receiving the OTP",
    "the password reset link is not working",
    "the app signs me out every time I log in",
    "I keep getting an invalid credentials message",
]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
    return text


def order_id(rng: random.Random) -> str:
    return f"SS{rng.randint(100000, 999999)}"


def transaction_id(rng: random.Random) -> str:
    return f"TXN{rng.randint(10000000, 99999999)}"


def sentence(*parts: str) -> str:
    return clean_text(" ".join(part for part in parts if part))


def prompt_with_style(rng: random.Random, body: str) -> str:
    intro = rng.choice(INTRO_PHRASES)
    ending = rng.choice(ENDINGS)
    customer = rng.choice(CUSTOMER_TYPES)
    return sentence(intro, body, customer, ending)


def choose(rng: random.Random, items: list[str]) -> str:
    return rng.choice(items)


def track_order_example(rng: random.Random) -> dict:
    product = choose(rng, PRODUCTS)
    body = choose(
        rng,
        [
            f"I placed an order for a {product} and I want to track where it is. My order ID is {order_id(rng)}.",
            f"Can you tell me the current status of order {order_id(rng)} for my {product}?",
            f"I ordered a {product} {choose(rng, TIME_PHRASES)} and I want to know how to track it.",
            f"I need a tracking update for order {order_id(rng)} because {choose(rng, ISSUES)}.",
        ],
    )
    response = choose(
        rng,
        [
            "Please open My Orders and use the tracking link to see the courier's latest scan. If the tracking page does not update for more than 48 hours, contact support with your order ID so the shipment can be escalated.",
            "You can track the shipment from the order details page in your account. If there is no movement for 48 hours or the tracking link is blank, share the order ID with support for a delivery investigation.",
            "The fastest way to check the package status is from the tracking section under your order details. If the status stays unchanged for more than 48 hours, contact support so the courier can be followed up.",
        ],
    )
    return {"category": "order_tracking", "prompt": prompt_with_style(rng, body), "response": response}


def delayed_delivery_example(rng: random.Random) -> dict:
    product = choose(rng, PRODUCTS)
    body = choose(
        rng,
        [
            f"My order for a {product} is delayed and {choose(rng, DELIVERY_WINDOWS)}.",
            f"Order {order_id(rng)} has not arrived yet even though {choose(rng, DELIVERY_WINDOWS)}.",
            f"I was expecting my {product} by now, but the parcel still has not arrived.",
            f"My shipment looks stuck in transit and {choose(rng, DELIVERY_WINDOWS)}.",
        ],
    )
    response = choose(
        rng,
        [
            "I am sorry your order is delayed. Please check the latest tracking update first, and if the parcel is still not delivered 48 hours after the promised date, contact support with your order ID so the delay can be escalated.",
            "Please review the courier tracking page from My Orders for the latest scan. If the order remains undelivered for 48 hours beyond the expected date, contact support so the shipment can be investigated.",
            "A short delay can happen in transit, but the order should be escalated if it is still pending 48 hours after the promised delivery date. Please share the order ID with support if that happens.",
        ],
    )
    return {"category": "delayed_delivery", "prompt": prompt_with_style(rng, body), "response": response}


def cancel_before_shipment_example(rng: random.Random) -> dict:
    product = choose(rng, PRODUCTS)
    body = choose(
        rng,
        [
            f"I want to cancel my order for the {product} before it ships.",
            f"Can I cancel order {order_id(rng)} right now? It still looks like it has not been shipped.",
            f"I placed a {product} order by mistake and I need to cancel it before dispatch.",
            f"My order is still in processing status. How do I cancel it?",
        ],
    )
    response = choose(
        rng,
        [
            "If the order status is still Confirmed or Processing, you can cancel it from My Orders. Once the cancellation is accepted, the refund starts automatically to the original payment method.",
            "You can cancel the order yourself from the order details page as long as it has not been shipped. After the cancellation is confirmed, the refund is initiated automatically.",
            "Cancellation is available only before shipment. Please open My Orders and tap cancel if the order is still in Processing or Confirmed status, and the refund will begin automatically after that.",
        ],
    )
    return {"category": "cancel_before_shipment", "prompt": prompt_with_style(rng, body), "response": response}


def cancel_after_shipment_example(rng: random.Random) -> dict:
    product = choose(rng, PRODUCTS)
    body = choose(
        rng,
        [
            f"My order for the {product} has already shipped, but I want to cancel it now.",
            f"Can you cancel order {order_id(rng)} even though the tracking page says it is shipped?",
            f"I no longer want the {product}, but it is already out for delivery.",
            f"The order is already in transit and I need to cancel it. What are my options?",
        ],
    )
    response = choose(
        rng,
        [
            "Once an order has been shipped, it cannot be canceled in the system. You can refuse the delivery or accept it and start a return request within 7 days of delivery if the item is eligible.",
            "Shipped orders cannot be canceled anymore. If you do not want the package, you can refuse delivery or place a return request within 7 days after delivery for eligible items.",
            "Cancellation is not available after shipment. Please refuse the parcel at delivery or submit a return request after delivery within the 7-day return window if the item is eligible.",
        ],
    )
    return {"category": "cancel_after_shipment", "prompt": prompt_with_style(rng, body), "response": response}


def return_request_example(rng: random.Random) -> dict:
    product = choose(rng, PRODUCTS)
    body = choose(
        rng,
        [
            f"I received my {product}, but {choose(rng, RETURN_REASONS)} and I want to return it.",
            f"How can I return order {order_id(rng)}? The item was delivered recently.",
            f"I want to place a return request for a {product} that was delivered {choose(rng, TIME_PHRASES)}.",
            f"Please tell me the return process for my {product}.",
        ],
    )
    response = choose(
        rng,
        [
            "You can request a return from My Orders within 7 days of delivery if the item is eligible and in its original condition. Please keep the product, tags, and packaging ready for pickup.",
            "Open the delivered order in My Orders and select Return to start the process. Returns are allowed within 7 days of delivery for eligible items that are unused and in original condition.",
            "Please raise the return request from your order details page within 7 days of delivery. Make sure the item is unused and packed with all original tags and accessories for a smooth pickup.",
        ],
    )
    return {"category": "return_request", "prompt": prompt_with_style(rng, body), "response": response}


def refund_status_example(rng: random.Random) -> dict:
    body = choose(
        rng,
        [
            f"I returned order {order_id(rng)} and I want to know when my refund will come.",
            "My return pickup is complete, but I still do not see the refund.",
            f"I am checking the refund status for a returned item. The payment method was {choose(rng, PAYMENT_METHODS)}.",
            "The return was approved a few days ago. How long does the refund take?",
        ],
    )
    response = choose(
        rng,
        [
            "Refunds are processed after the returned item passes inspection. For prepaid orders, the amount usually reaches the original payment method within 5 to 7 business days, while COD refunds are sent to the selected bank account or UPI ID within 3 to 5 business days after approval.",
            "Once the return is verified, the refund is started automatically. Prepaid refunds usually take 5 to 7 business days to reflect, and COD refunds are completed to the registered bank or UPI details within 3 to 5 business days after approval.",
            "Please allow the return quality check to finish first. After approval, prepaid refunds normally take 5 to 7 business days, and COD refunds take about 3 to 5 business days to the bank account or UPI details provided.",
        ],
    )
    return {"category": "refund_status", "prompt": prompt_with_style(rng, body), "response": response}


def exchange_request_example(rng: random.Random) -> dict:
    product = choose(rng, PRODUCTS)
    current_size = choose(rng, SIZES)
    new_size = choose(rng, SIZES)
    current_color = choose(rng, COLORS)
    new_color = choose(rng, COLORS)
    body = choose(
        rng,
        [
            f"I ordered a {product} in size {current_size}, but I need size {new_size}. Can I exchange it?",
            f"I want to exchange my {current_color} {product} for {new_color}.",
            f"My {product} was delivered, but I need a different size or color. How do I exchange it?",
            f"Can you help me exchange order {order_id(rng)} for another variant of the same item?",
        ],
    )
    response = choose(
        rng,
        [
            "You can request an exchange from My Orders within 7 days of delivery if exchange is available for that product. Exchanges are completed only when the requested size or color is in stock; otherwise you can return the item for a refund.",
            "Please open the delivered order and check whether Exchange is available for that item. If the new variant is in stock, the exchange can be placed within 7 days of delivery, and if not, you can return the item instead.",
            "Exchange requests are allowed within 7 days of delivery for eligible products and only when the requested variant is available. If the size or color is out of stock, please place a return request for a refund.",
        ],
    )
    return {"category": "exchange_request", "prompt": prompt_with_style(rng, body), "response": response}


def wrong_item_example(rng: random.Random) -> dict:
    expected = choose(rng, PRODUCTS)
    received = choose(rng, PRODUCTS)
    body = choose(
        rng,
        [
            f"I ordered a {expected}, but I received a {received} instead.",
            f"Order {order_id(rng)} contains the wrong item. What should I do?",
            f"The package delivered today has a different product than what I purchased.",
            f"I received the wrong variant in my order and I need this fixed.",
        ],
    )
    response = choose(
        rng,
        [
            "I am sorry you received the wrong item. Please report it within 48 hours of delivery from My Orders and upload clear photos of the product, packaging, and shipping label so a replacement or refund can be arranged.",
            "Please raise a Wrong Item issue within 48 hours of delivery and attach photos of the received product, outer package, and label. Once verified, support can arrange a replacement or refund.",
            "If the delivered item does not match the order, report the issue within 48 hours from the order details page. Clear photos of the item and package are needed so the case can be verified for replacement or refund.",
        ],
    )
    return {"category": "wrong_item_received", "prompt": prompt_with_style(rng, body), "response": response}


def damaged_item_example(rng: random.Random) -> dict:
    product = choose(rng, PRODUCTS)
    body = choose(
        rng,
        [
            f"My {product} arrived damaged today.",
            f"Order {order_id(rng)} was delivered, but the item is broken.",
            f"I opened my package and the {product} is damaged. How do I report this?",
            f"The product I received has visible damage and I need a replacement or refund.",
        ],
    )
    response = choose(
        rng,
        [
            "I am sorry the item arrived damaged. Please report the damage within 48 hours of delivery and upload clear photos or an unboxing video from My Orders so the case can be verified for a replacement or refund.",
            "Please open the order details page and submit a Damaged Item request within 48 hours of delivery. Clear product photos, package photos, and any unboxing video will help support process the replacement or refund faster.",
            "Damage issues must be reported within 48 hours of delivery from the order page. Please attach photos of the product and packaging, and include an unboxing video if available, so the request can be reviewed for refund or replacement.",
        ],
    )
    return {"category": "damaged_item", "prompt": prompt_with_style(rng, body), "response": response}


def missing_item_example(rng: random.Random) -> dict:
    product = choose(rng, PRODUCTS)
    body = choose(
        rng,
        [
            f"My package was delivered, but the {product} is missing from the box.",
            f"Order {order_id(rng)} arrived with one item missing.",
            f"I received the parcel today, but one product is not inside the package.",
            f"The box looked incomplete and my item is missing. What should I do next?",
        ],
    )
    response = choose(
        rng,
        [
            "Please report the missing item within 48 hours of delivery from My Orders. Upload photos of the outer package, shipping label, and everything that was inside the box so the shortage can be investigated.",
            "If an item is missing from a delivered order, raise the issue within 48 hours and attach clear photos of the package, label, and received contents. Support will review the case and arrange a refund or replacement if it is verified.",
            "A missing item complaint should be submitted within 48 hours of delivery from the order details page. Please share package photos and the shipping label so the logistics team can investigate the shortage.",
        ],
    )
    return {"category": "missing_item", "prompt": prompt_with_style(rng, body), "response": response}


def payment_failed_example(rng: random.Random) -> dict:
    method = choose(rng, PAYMENT_METHODS)
    body = choose(
        rng,
        [
            f"My payment failed while using {method}, and the order did not go through.",
            f"I was trying to place an order, but the checkout failed at the payment step.",
            f"The app showed a payment failure message when I tried to pay by {method}.",
            "I cannot complete my purchase because the payment page keeps failing.",
        ],
    )
    response = choose(
        rng,
        [
            "Please try placing the order again after checking your payment details and internet connection, or switch to another payment method. If no money was deducted and no order was created, the failed attempt does not need any further action.",
            "A failed payment usually means the order was not placed. Please retry the checkout once, and if the issue continues, use another payment option or contact your bank or payment provider.",
            "If the payment failed and no amount was deducted, you can safely retry the order or use a different payment method. If the error keeps repeating, contact support with a screenshot of the checkout error.",
        ],
    )
    return {"category": "payment_failed", "prompt": prompt_with_style(rng, body), "response": response}


def payment_deducted_example(rng: random.Random) -> dict:
    method = choose(rng, PAYMENT_METHODS[:-1])
    body = choose(
        rng,
        [
            f"My payment was deducted through {method}, but no order was created.",
            f"I paid for checkout, but the app crashed and I cannot see any order. The transaction ID is {transaction_id(rng)}.",
            "Money has been debited from my account, but I never got an order confirmation.",
            f"The payment went through, but the order page is empty. What should I do now?",
        ],
    )
    response = choose(
        rng,
        [
            "If the payment was deducted but the order was not created, the amount is usually reversed automatically within 3 to 5 business days. If the refund is not received after that, contact support with the payment reference or bank transaction ID.",
            "Please wait for 3 to 5 business days because most failed checkout deductions are auto-reversed by the bank or payment gateway. If the amount is still not credited back after that, share the transaction ID with support for manual review.",
            "A deduction without order confirmation is normally refunded automatically within 3 to 5 business days. If there is no reversal after that period, contact support with the transaction details so the payment team can investigate.",
        ],
    )
    return {"category": "payment_deducted_no_order", "prompt": prompt_with_style(rng, body), "response": response}


def address_change_example(rng: random.Random) -> dict:
    body = choose(
        rng,
        [
            f"I entered the wrong shipping address for order {order_id(rng)}. How can I change it?",
            "I want to update my delivery address before the order ships.",
            "The pin code on my order is wrong and I need to correct the address.",
            "Can I edit the address for my current order? It still looks like it has not been dispatched.",
        ],
    )
    response = choose(
        rng,
        [
            "You can change the delivery address only before the order is shipped. Please check My Orders for the edit option, and if the order is already packed or shipped, the address can no longer be changed.",
            "Address updates are allowed only until shipment. Open the order details page and edit the address if that option is visible; once the order is shipped, the delivery address cannot be modified.",
            "Please try updating the address from My Orders while the order is still in Confirmed or Processing status. If the shipment has already started, the address cannot be changed anymore.",
        ],
    )
    return {"category": "address_change", "prompt": prompt_with_style(rng, body), "response": response}


def coupon_issue_example(rng: random.Random) -> dict:
    coupon = choose(rng, COUPONS)
    body = choose(
        rng,
        [
            f"The coupon code {coupon} is not working on my cart.",
            f"I am trying to use {coupon}, but checkout says the coupon is invalid.",
            f"My discount code is failing even though I typed it correctly.",
            f"Why is the coupon not applying to order {order_id(rng)}?",
        ],
    )
    response = choose(
        rng,
        [
            "Please check whether the coupon is still valid, meets the minimum cart value, and is eligible for the products in your cart. Only one coupon can be used per order, and some brands or categories may be excluded from discounts.",
            "Coupon errors usually happen when the code is expired, the cart value is below the minimum requirement, or the items are excluded from offers. Please also make sure only one coupon is being applied to the order.",
            "Please verify the coupon expiry, minimum spend, and product eligibility first. Discount codes cannot be combined, and some sellers or categories are excluded from promotional offers.",
        ],
    )
    return {"category": "coupon_not_working", "prompt": prompt_with_style(rng, body), "response": response}


def warranty_claim_example(rng: random.Random) -> dict:
    product = choose(rng, PRODUCTS)
    body = choose(
        rng,
        [
            f"My {product} has stopped working and I want to claim warranty.",
            f"How do I start a warranty claim for order {order_id(rng)}?",
            f"The product was delivered earlier, but now it has developed a fault. What is the warranty process?",
            f"I need support for a warranty issue with my {product}.",
        ],
    )
    response = choose(
        rng,
        [
            "Please contact support with your order ID, a short description of the fault, and photos or video if the issue is visible. Warranty claims are handled based on the seller or brand policy, and the team will guide you on repair, replacement, or service center steps.",
            "To begin a warranty claim, share the order ID, product issue, and any supporting photos or video with support. The next step depends on the seller or brand warranty terms, and you will be guided on repair or replacement options.",
            "Please keep the invoice and order ID ready and contact support with a clear description of the issue. If the defect is visible, include photos or video so the warranty process can be reviewed under the brand or seller policy.",
        ],
    )
    return {"category": "warranty_claim", "prompt": prompt_with_style(rng, body), "response": response}


def out_of_stock_example(rng: random.Random) -> dict:
    product = choose(rng, PRODUCTS)
    body = choose(
        rng,
        [
            f"The {product} I want to buy is out of stock. When will it be available again?",
            f"Can you tell me if the {product} will be restocked soon?",
            f"I am waiting for a {product}, but it still shows out of stock.",
            f"Do you reserve stock for customers, or should I wait for the item to come back?",
        ],
    )
    response = choose(
        rng,
        [
            "Restock timing depends on the seller and current inventory, so support cannot promise a fixed date. Please use the Notify Me option on the product page so you receive an alert if the item becomes available again.",
            "Out-of-stock items are restocked based on seller inventory updates, so there is no guaranteed date. Please enable stock alerts on the product page and check back later for availability.",
            "We cannot confirm a restock date in advance because availability depends on the seller. The best option is to turn on Notify Me for that product so you are informed when the item returns.",
        ],
    )
    return {"category": "out_of_stock_query", "prompt": prompt_with_style(rng, body), "response": response}


def account_login_example(rng: random.Random) -> dict:
    body = choose(
        rng,
        [
            f"I cannot log in to my account because {choose(rng, LOGIN_ISSUES)}.",
            "I am locked out of my shopping account and need help getting back in.",
            "The app is not letting me sign in even though I am using the correct details.",
            "How do I recover access to my account if login is failing?",
        ],
    )
    response = choose(
        rng,
        [
            "Please use Forgot Password or the OTP login option first and make sure you are using the registered email address or phone number. If you still cannot log in, clear the app cache, try again, and contact support if the issue continues.",
            "Start with password reset or OTP login using your registered contact details. If the OTP does not arrive or the app keeps failing, clear cache or try the website version before contacting support for account assistance.",
            "Please confirm that you are using the correct registered email or phone number, then try the reset password or OTP option. If login still fails after clearing cache or trying another browser, contact support for further verification.",
        ],
    )
    return {"category": "account_login_issue", "prompt": prompt_with_style(rng, body), "response": response}


def invoice_request_example(rng: random.Random) -> dict:
    product = choose(rng, PRODUCTS)
    body = choose(
        rng,
        [
            f"I need the invoice for my {product} order.",
            f"How can I download the bill for order {order_id(rng)}?",
            "I need a GST invoice or order invoice for reimbursement.",
            "The item was delivered, but I cannot find the invoice in my account.",
        ],
    )
    response = choose(
        rng,
        [
            "You can download the invoice from My Orders once the item is shipped or delivered. Open the order details page and look for the Invoice option, and if it is still missing after 24 hours, contact support with the order ID.",
            "Please check the order details page in My Orders for the invoice download option. If the invoice is not visible within 24 hours after shipment or delivery, contact support so it can be reviewed.",
            "Invoices are usually available from the order details page after shipment or delivery. If you still do not see the invoice after 24 hours, share the order ID with support for help.",
        ],
    )
    return {"category": "invoice_request", "prompt": prompt_with_style(rng, body), "response": response}


GENERATORS = [
    track_order_example,
    delayed_delivery_example,
    cancel_before_shipment_example,
    cancel_after_shipment_example,
    return_request_example,
    refund_status_example,
    exchange_request_example,
    wrong_item_example,
    damaged_item_example,
    missing_item_example,
    payment_failed_example,
    payment_deducted_example,
    address_change_example,
    coupon_issue_example,
    warranty_claim_example,
    out_of_stock_example,
    account_login_example,
    invoice_request_example,
]


def generate_examples() -> list[dict]:
    rng = random.Random(SEED)
    examples: list[dict] = []

    for generator in GENERATORS:
        category_examples: list[dict] = []
        seen_prompts: set[str] = set()
        while len(category_examples) < EXAMPLES_PER_CATEGORY:
            example = generator(rng)
            prompt_key = example["prompt"].lower()
            if prompt_key in seen_prompts:
                continue
            seen_prompts.add(prompt_key)
            category_examples.append(example)
        examples.extend(category_examples)

    return examples


def split_examples(examples: list[dict]) -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        grouped[example["category"]].append(example)

    splits = {"train": [], "validation": [], "test": []}
    example_id = 1

    for category in sorted(grouped):
        items = grouped[category][:]
        rng.shuffle(items)

        train_items = items[:TRAIN_PER_CATEGORY]
        val_items = items[TRAIN_PER_CATEGORY : TRAIN_PER_CATEGORY + VAL_PER_CATEGORY]
        test_items = items[TRAIN_PER_CATEGORY + VAL_PER_CATEGORY :]

        for split_name, split_items in [("train", train_items), ("validation", val_items), ("test", test_items)]:
            for item in split_items:
                split_record = {
                    "id": f"ecs_{example_id:04d}",
                    "dataset_name": "ecommerce_customer_support_assistant",
                    "domain": "e-commerce customer support",
                    "store_name": STORE_NAME,
                    "split": split_name,
                    **item,
                }
                splits[split_name].append(split_record)
                example_id += 1

    splits["all"] = splits["train"] + splits["validation"] + splits["test"]
    return splits


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_stats(path: Path, splits: dict[str, list[dict]]) -> None:
    stats = {
        "dataset_name": "ecommerce_customer_support_assistant",
        "store_name": STORE_NAME,
        "total_examples": len(splits["all"]),
        "split_counts": {name: len(splits[name]) for name in ["train", "validation", "test"]},
        "category_counts": dict(Counter(row["category"] for row in splits["all"])),
    }
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    examples = generate_examples()
    splits = split_examples(examples)

    for split_name in ["all", "train", "validation", "test"]:
        rows = splits[split_name]
        write_jsonl(OUTPUT_DIR / f"{split_name}.jsonl", rows)
        write_csv(OUTPUT_DIR / f"{split_name}.csv", rows)

    write_stats(OUTPUT_DIR / "stats.json", splits)

    print(f"Generated dataset at: {OUTPUT_DIR}")
    print(f"Total examples: {len(splits['all'])}")
    print(f"Train: {len(splits['train'])} | Validation: {len(splits['validation'])} | Test: {len(splits['test'])}")
    print("Categories:")
    for category, count in sorted(Counter(row["category"] for row in splits["all"]).items()):
        print(f"  - {category}: {count}")


if __name__ == "__main__":
    main()
