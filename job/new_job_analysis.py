import argparse
from datetime import datetime,timedelta

def days_to_recover_loss(d1, d2, s1, s2):
    # Convert strings to datetime objects
    leave_date = datetime.strptime(d1, '%Y-%m-%d')
    join_date = datetime.strptime(d2, '%Y-%m-%d')

    # Calculate unemployment duration in days
    unemployment_days = (join_date - leave_date).days

    # Calculate lost earnings during unemployment
    daily_last_salary = s1 / 365
    lost_earnings = daily_last_salary * unemployment_days

    # Calculate daily salary in new company
    daily_new_salary = s2 / 365

    # Solve for N where N * (new_salary - last_salary) / 365 = lost_earnings
    if daily_new_salary > daily_last_salary:
        required_days = lost_earnings / (daily_new_salary - daily_last_salary)
    else:
        return "New salary must be greater than the last salary to recover the loss."

    final_date = join_date + timedelta(days=int(required_days))
    earnings_if_stayed = daily_last_salary * (final_date - leave_date).days
    earnings_in_new_job = daily_new_salary * (final_date - join_date).days

    print(f"From {leave_date.date()} to {final_date.date()}, you would have earned ${earnings_if_stayed:,.0f} in your old job.")
    print(f"From {join_date.date()} to {final_date.date()}, you will earn ${earnings_in_new_job:,.0f} in {int(required_days)} days at your new job.")

    return int(required_days)  # Return the number of days needed to recover loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate days needed to recover job loss earnings.")
    parser.add_argument("-d1", type=str, default='2024-11-14', help="Leave date (YYYY-MM-DD)")
    parser.add_argument("-d2", type=str, default='2025-09-01', help="Join date (YYYY-MM-DD)")
    parser.add_argument("-s1", type=int, default=111000, help="Last job salary per year")
    parser.add_argument("-s2", type=int, default=130000, help="New job salary per year")

    args = parser.parse_args()

    days_needed = days_to_recover_loss(args.d1, args.d2, args.s1, args.s2)
    print(f"Number of days needed to recover the loss: {days_needed}")
