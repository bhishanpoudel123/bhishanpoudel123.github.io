
# Date: 2025 Mar 27
import numpy as np

def calculate_stock_average(purchases, current_price, target_price):
    """
    Calculates the number of shares required to reach the target average price.

    :param purchases: List of tuples (shares, price) representing past purchases.
    :param current_price: The current stock price.
    :param target_price: The target average stock price.
    :return: The number of shares needed to buy and total cost.
    """
    total_shares = sum(shares for shares, _ in purchases)
    total_cost = sum(shares * price for shares, price in purchases)

    if total_shares == 0 or current_price <= 0 or target_price <= 0:
        return "Invalid input values!"

    required_shares = (total_cost - (target_price * total_shares)) / (target_price - current_price)

    if required_shares < 0:
        return "Target price is already met or lower than current price!"

    required_shares = int(required_shares) + 1  # Round up to ensure target price is reached
    required_cost = required_shares * current_price

    return required_shares, round(required_cost, 2)


# Example test case
purchases = [(2700, 6.34), (2800, 6.90)]  # (shares, price) from Robinhood and Webull
current_price = 4.00
target_price = 5.00

result = calculate_stock_average(purchases, current_price, target_price)
print(f"You need to buy: {result[0]} shares")
print(f"Total Cost: ${result[1]}")

answer = r"""

You need to buy: 8939 shares
Total Cost: $35756.0

"""