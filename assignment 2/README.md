# Assignment 2: Fine-Tuning Dataset

This folder contains a custom synthetic dataset for the task `e-commerce customer support assistant`.

## Dataset Summary

- Domain: e-commerce customer support
- Dataset style: prompt-response supervised fine-tuning
- Store name used in examples: `ShopSphere`
- Total examples: `720`
- Splits:
  - train: `576`
  - validation: `72`
  - test: `72`
- Categories: `18`

## Categories

- order tracking
- delayed delivery
- cancel before shipment
- cancel after shipment
- return request
- refund status
- exchange request
- wrong item received
- damaged item
- missing item
- payment failed
- payment deducted but no order created
- address change
- coupon not working
- warranty claim
- out of stock query
- account login issue
- invoice request

## Dataset Format

Each row contains:

- `id`
- `dataset_name`
- `domain`
- `store_name`
- `split`
- `category`
- `prompt`
- `response`

Example:

```json
{
  "id": "ecs_0001",
  "dataset_name": "ecommerce_customer_support_assistant",
  "domain": "e-commerce customer support",
  "store_name": "ShopSphere",
  "split": "train",
  "category": "refund_status",
  "prompt": "Hi, I returned order SS482193 and I want to know when my refund will come. Please help.",
  "response": "Refunds are processed after the returned item passes inspection. For prepaid orders, the amount usually reaches the original payment method within 5 to 7 business days, while COD refunds are sent to the selected bank account or UPI ID within 3 to 5 business days after approval."
}
```

## Policy Assumptions Used In Responses

The target replies are consistent with a fixed store policy:

- cancellations are allowed only before shipment
- eligible returns and exchanges are allowed within 7 days of delivery
- wrong, damaged, or missing item issues should be reported within 48 hours of delivery
- prepaid refunds usually take 5 to 7 business days after approval
- COD refunds usually take 3 to 5 business days after approval
- address changes are allowed only before shipment
- only one coupon can be used per order

## Files

- `data/ecommerce_support/all.jsonl`
- `data/ecommerce_support/train.jsonl`
- `data/ecommerce_support/validation.jsonl`
- `data/ecommerce_support/test.jsonl`
- `data/ecommerce_support/all.csv`
- `data/ecommerce_support/train.csv`
- `data/ecommerce_support/validation.csv`
- `data/ecommerce_support/test.csv`
- `data/ecommerce_support/stats.json`
- `scripts/generate_dataset.py`

## Regeneration

Run this from `assignment 2`:

```bash
python3 scripts/generate_dataset.py
```
