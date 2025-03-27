import argparse

def calculate_stock_average(purchases, current_price, target_price):
    """
    Calculates the number of shares required to reach the target average price.

    :param purchases: List of tuples (shares, price) representing past purchases.
    :param current_price: The current stock price.
    :param target_price: The target average stock price.
    :return: Dictionary with calculation results.
    """
    total_shares = sum(shares for shares, _ in purchases)
    total_cost = sum(shares * price for shares, price in purchases)

    if total_shares == 0 or current_price <= 0 or target_price <= 0:
        return {"error": "Invalid input values!"}

    avg_cost = total_cost / total_shares
    required_shares = (total_cost - (target_price * total_shares)) / (target_price - current_price)

    if required_shares < 0:
        return {"error": "Target price is already met or lower than current price!"}

    required_shares = int(required_shares) + 1  # Round up
    required_cost = required_shares * current_price
    percent_increase_needed = ((avg_cost - current_price) / current_price) * 100

    return {
        "required_shares": required_shares,
        "required_cost": round(required_cost, 2),
        "avg_cost": round(avg_cost, 2),
        "percent_increase_needed": round(percent_increase_needed, 2)
    }

def main():
    parser = argparse.ArgumentParser(description="Stock Average Calculator")

    parser.add_argument("-p", "--purchases", nargs="+", type=str,
                        default=["2700:6.34", "2800:6.90"],required=True,
                        help="List of purchases in format 'shares:price' (e.g., '2700:6.34')")
    parser.add_argument("-c", "--current_price", type=float, default=4.00, help="Current stock price",required=True)
    parser.add_argument("-t", "--target_price", type=float, default=5.00, help="Target average price",required=True)

    args = parser.parse_args()

    # Parse purchases input
    purchases = []
    for purchase in args.purchases:
        try:
            shares, price = map(float, purchase.split(":"))
            purchases.append((int(shares), price))
        except ValueError:
            print(f"Invalid purchase format: {purchase}. Use 'shares:price' format.")
            return

    # Perform calculation
    result = calculate_stock_average(purchases, args.current_price, args.target_price)

    # Output results
    if "error" in result:
        print(result["error"])
    else:
        print(f"You need to buy: {result['required_shares']} shares")
        print(f"Total Cost: ${result['required_cost']}")
        print(f"Overall Stock Average Price: ${result['avg_cost']}")
        print(f"Percent Increase Needed: {result['percent_increase_needed']}%")

if __name__ == "__main__":
    main()
