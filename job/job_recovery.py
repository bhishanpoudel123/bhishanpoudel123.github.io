#!/usr/bin/env python3
import argparse
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

# --- Utility functions ---
def days_between(start_date, end_date):
    return (end_date - start_date).days

def convert_days_to_ymd(days):
    start = date(2000, 1, 1)
    end = start + timedelta(days=days)
    years = end.year - start.year
    months = end.month - start.month
    extra_days = end.day - start.day

    if extra_days < 0:
        months -= 1
        prev_month = (end.month - 1) if end.month > 1 else 12
        prev_year = end.year if end.month > 1 else end.year - 1
        extra_days += (date(prev_year, prev_month + 1, 1) - timedelta(days=1)).day

    if months < 0:
        years -= 1
        months += 12

    return f"{years}y {months}m {extra_days}d"

def calculate_annual_earnings(start_date, end_date, base_salary, annual_increase, verbose=False, label=""):
    total_earnings = 0
    current_date = start_date
    current_salary = base_salary
    period_count = 1

    while current_date < end_date:
        total_earnings += current_salary / 365
        next_date = current_date + timedelta(days=1)
        current_date = next_date
        if current_date.month == 10 and current_date.day == 1:
            current_salary *= (1 + annual_increase / 100)
            if verbose:
                print(f"{label} raise applied on {current_date}: ${current_salary:,.2f}")
            period_count += 1
    return round(total_earnings)

def calculate_recovery_time_compound(lost_earnings, new_salary, last_salary, join_date, old_increase, new_increase, max_days=3650):
    cumulative_difference = 0.0
    days_count = 0
    current_date = join_date
    current_old_salary = last_salary
    current_new_salary = new_salary

    while cumulative_difference < lost_earnings and days_count < max_days:
        daily_old = current_old_salary / 365.0
        daily_new = current_new_salary / 365.0
        cumulative_difference += daily_new - daily_old
        days_count += 1
        current_date += timedelta(days=1)

        if current_date.month == 10 and current_date.day == 1:
            current_old_salary *= (1 + old_increase / 100)
            current_new_salary *= (1 + new_increase / 100)

    return {
        "days_count": days_count,
        "total_amount": round(cumulative_difference),
        "daily_old": daily_old,
        "daily_new": daily_new,
        "cum_old": daily_old * days_count,
        "cum_new": daily_new * days_count,
        "breakeven_date": current_date
    }

def project_salary_for_year(leave_date, join_date, last_salary, new_salary,
                            old_increase, new_increase, target_year, verbose=False):
    end_date = date(target_year, leave_date.month, leave_date.day)
    old_earnings = calculate_annual_earnings(leave_date, end_date, last_salary, old_increase, verbose=verbose, label="Old Job")
    new_earnings = 0
    if end_date > join_date:
        new_earnings = calculate_annual_earnings(join_date, end_date, new_salary, new_increase, verbose=verbose, label="New Job")
    return old_earnings, new_earnings, new_earnings - old_earnings

def format_breakeven_info(breakeven_info, leave_date, join_date):
    today = date.today()
    delta_today = relativedelta(breakeven_info["breakeven_date"], today)
    delta_leave = relativedelta(breakeven_info["breakeven_date"], leave_date)
    delta_join = relativedelta(breakeven_info["breakeven_date"], join_date)

    output = f"""
=== Breakeven / Closest Day Analysis ===
Status: {'Breakeven (New ≥ Old)' if breakeven_info['total_amount'] >= 0 else 'Closest (New < Old)'}
Date: {breakeven_info['breakeven_date'].strftime('%Y-%m-%d')}
Cumulative Old Job Earnings: ${breakeven_info['cum_old']:,.2f}
Cumulative New Job Earnings: ${breakeven_info['cum_new']:,.2f}
Gap (New - Old): ${breakeven_info['total_amount']:,.2f}

Daily Salary Rates:
  Old Job: ${breakeven_info['daily_old']:,.2f}/day
  New Job: ${breakeven_info['daily_new']:,.2f}/day

Annual Salaries:
  Old Job: ${breakeven_info['daily_old']*365:,.0f}/year
  New Job: ${breakeven_info['daily_new']*365:,.0f}/year

Time from leave date ({leave_date.strftime('%Y-%m-%d')}): {delta_leave.years}y {delta_leave.months}m {delta_leave.days}d
Time from join date ({join_date.strftime('%Y-%m-%d')}): {delta_join.years}y {delta_join.months}m {delta_join.days}d
Time from today ({today.strftime('%Y-%m-%d')}): {delta_today.years}y {delta_today.months}m {delta_today.days}d
"""
    return output

# --- Main CLI ---
def main():
    parser = argparse.ArgumentParser(description="Job Loss Recovery Salary Projection Calculator")
    parser.add_argument("-l", "--leave-date", type=str, default="2024-11-14")
    parser.add_argument("-j", "--join-date", type=str, default="2025-09-01")
    parser.add_argument("-s", "--last-salary", type=float, default=111000)
    parser.add_argument("-n", "--new-salary", type=float, default=130000)
    parser.add_argument("-o", "--old-increase", type=float, default=3, help="Old job annual salary increase percentage")
    parser.add_argument("-N", "--new-increase", type=float, default=3, help="New job annual salary increase percentage")
    parser.add_argument("-y", "--year", type=int, default=2025)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    leave_date = date.fromisoformat(args.leave_date)
    join_date = date.fromisoformat(args.join_date)
    unemployment_days = days_between(leave_date, join_date)
    daily_last_salary = args.last_salary / 365.0
    lost_earnings = round(daily_last_salary * unemployment_days)

    # Recovery calculation
    breakeven_info = calculate_recovery_time_compound(
        lost_earnings, args.new_salary, args.last_salary,
        join_date, args.old_increase, args.new_increase
    )

    cover_time = convert_days_to_ymd(breakeven_info["days_count"])

    # Print Recovery Summary
    print("\n=== Recovery Summary ===")
    print(f"Unemployment period: {convert_days_to_ymd(unemployment_days)} ({unemployment_days} days)")
    print(f"Lost earnings: ${lost_earnings:,.0f}")
    print(f"Time to break even: {cover_time}")
    print(f"Total recovery amount: ${breakeven_info['total_amount']:,.0f}")

    # Projection for target year
    old_earnings, new_earnings, diff = project_salary_for_year(
        leave_date, join_date, args.last_salary, args.new_salary,
        args.old_increase, args.new_increase, args.year, verbose=args.verbose
    )
    target_end_date = date(args.year, leave_date.month, leave_date.day)
    print(f"\n=== Projection for {target_end_date.strftime('%Y-%m-%d')} ===")
    print(f"Old job earnings: ${old_earnings:,.0f}")
    print(f"New job earnings: ${new_earnings:,.0f}")
    print(f"Difference: ${diff:,.0f}")

    # Human-readable breakeven info
    readable_output = format_breakeven_info(breakeven_info, leave_date, join_date)
    print(readable_output)

if __name__ == "__main__":
    main()
