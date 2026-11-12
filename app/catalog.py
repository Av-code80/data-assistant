"""Semantic catalog: the business metrics and dimensions the assistant can answer about.

Single source of truth. The same lists feed the Pydantic types (so the LLM can only
pick known values) and the system prompt (so the LLM knows what each one means).
The business glossary in knowledge/ explains them in the company's own words (RAG).
"""

METRICS: dict[str, str] = {
    "revenue": "Sum of item sale prices for orders that were not cancelled or returned.",
    "order_count": "Number of distinct orders that were not cancelled.",
    "average_order_value": "Revenue divided by order_count.",
    "units_sold": "Number of items sold in orders that were not cancelled or returned.",
    "customer_count": "Number of distinct customers who placed at least one order.",
    "return_rate": "Returned items divided by items sold, as a percentage.",
}

DIMENSIONS: dict[str, str] = {
    "product_category": "Product category, e.g. Jeans, Sweaters, Accessories.",
    "brand": "Product brand.",
    "department": "Product department, who the product is designed for: Women, Men. "
                  "Use it for \"pour femme\", \"rayon homme\", \"mode masculine\".",
    "country": "Customer country.",
    "city": "Customer city.",
    "traffic_source": "Acquisition channel: Search, Organic, Email, Facebook, Display.",
    "gender": "Gender of the customer who bought: F, M. Use it for \"les clientes\", "
              "\"achats des hommes\", not for the product's department.",
    "age_group": "Customer age bucket: 18-24, 25-34, 35-44, 45-54, 55+.",
    "month": "Calendar month of the order.",
    "week": "Calendar week of the order.",
    "day": "Calendar day of the order.",
}

METRIC_NAMES = tuple(METRICS)
DIMENSION_NAMES = tuple(DIMENSIONS)


def render_catalog() -> str:
    """Format the catalog as plain text for the system prompt."""
    metrics = "\n".join(f"- {name}: {desc}" for name, desc in METRICS.items())
    dimensions = "\n".join(f"- {name}: {desc}" for name, desc in DIMENSIONS.items())
    return f"Metrics:\n{metrics}\n\nDimensions:\n{dimensions}"
