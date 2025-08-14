#!/usr/bin/env python3
import argparse
from datetime import date, timedelta

def days_between(start_date, end_date):
    start = date(start_date.year, start_date.month, start_date.day)
    end = date(end_date.year, end_date.month, end_date.day)
    return (end - start).days

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
    """
    Day-accurate earnings with raises applied each Oct 1 *after* that day's pay,
    matching the browser logic.
    """
    if start_date >= end_date:
        return 0

    total_earnings = 0
    current_date = date(start_date.year, start_date.month, start_date.day)
    current_salary = base_salary
    period_count = 1

    if verbose:
        print(f"\\n--- {label} Calculation from {start_date} to {end_date} ---")
        print(f"Starting salary: ${current_salary:,.2f} | Annual increase: {annual_increase}%")

    # Iterate day-by-day so the Oct 1 raise happens *after* that day's accrual
    while current_date < end_date:
        # accrue one day's pay
        total_earnings += current_salary / 365
        next_date = current_date + timedelta(days=1)

        if verbose and (current_date.day == 1 and current_date.month == 10):
            # This message reflects pre-raise accrual for Oct 1 then raise for next day
            pass

        # Move to next day
        current_date = next_date

        # Apply raise if we've moved into Oct 1
        if current_date.month == 10 and current_date.day == 1:
            current_salary *= (1 + annual_increase / 100)
            if verbose:
                print(f"Period {period_count}: raise on {current_date} → new salary ${current_salary:,.2f}")
                period_count += 1

    return round(total_earnings)

def calculate_recovery_time_compound(lost_earnings, new_salary, last_salary, join_date, old_increase, new_increase, max_days=3650):
    """
    Compound recovery calculation mirroring index.html:
    - Compare daily rates (new vs old)
    - Apply raises each Oct 1 *after* that day's accrual
    - Stop when cumulative difference covers lost_earnings or max_days hit
    """
    cumulative_difference = 0.0
    days_count = 0

    current_date = join_date
    current_old_salary = last_salary
    current_new_salary = new_salary

    while cumulative_difference < lost_earnings and days_count < max_days:
        daily_old = current_old_salary / 365.0
        daily_new = current_new_salary / 365.0
        daily_diff = daily_new - daily_old
        cumulative_difference += daily_diff
        days_count += 1

        # advance a day
        current_date = current_date + timedelta(days=1)

        # apply raises when the *next* day becomes Oct 1
        if current_date.month == 10 and current_date.day == 1:
            current_old_salary *= (1 + old_increase / 100.0)
            current_new_salary *= (1 + new_increase / 100.0)

    return {
        "days_count": days_count,
        "total_amount": round(cumulative_difference),
    }

def project_salary_for_year(leave_date, join_date, last_salary, new_salary,
                            old_increase, new_increase, target_year, verbose=False):
    """
    Project cumulative earnings through the same calendar day-month of leave_date
    in target_year for both jobs, using day-accurate accrual and Oct 1 raises.
    """
    end_date = date(target_year, leave_date.month, leave_date.day)
    old_earnings = calculate_annual_earnings(leave_date, end_date, last_salary, old_increase, verbose=verbose, label="Old Job")
    new_earnings = 0
    if end_date > join_date:
        new_earnings = calculate_annual_earnings(join_date, end_date, new_salary, new_increase, verbose=verbose, label="New Job")
    return old_earnings, new_earnings, new_earnings - old_earnings

def main():
    description = """
Job Loss Recovery Salary Projection Calculator (compound version)

This tool mirrors the browser logic exactly:
- Break-even (recovery) time includes *compounded* raises for both jobs each Oct 1
- Earnings are day-accurate across year boundaries
- Projections show old vs new cumulative earnings with the same rules

Example usage:
  python job_recovery.py --year 2027
  python job_recovery.py -l 2024-11-14 -j 2025-09-01 -s 111000 -n 140000 -o 3 -N 2 -y 2028
"""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("-l", "--leave-date", type=str, default="2024-11-14",
                        help="Last leave date from old job (YYYY-MM-DD)")
    parser.add_argument("-j", "--join-date", type=str, default="2025-09-01",
                        help="Start date of new job (YYYY-MM-DD)")
    parser.add_argument("-s", "--last-salary", type=float, default=111000,
                        help="Annual salary at old job before leaving")
    parser.add_argument("-n", "--new-salary", type=float, default=130000,
                        help="Annual salary at new job upon joining")
    parser.add_argument("-o", "--old-increase", type=float, default=3,
                        help="Old job annual salary increase percentage")
    parser.add_argument("-N", "--new-increase", type=float, default=0,
                        help="New job annual salary increase percentage")
    parser.add_argument("-y", "--year", type=int, default=2025,
                        help="Target year for detailed projection (YYYY)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print detailed accrual and raise events for projections")

    args = parser.parse_args()

    leave_date = date.fromisoformat(args.leave_date)
    join_date = date.fromisoformat(args.join_date)

    # Unemployment loss based on old job's daily rate
    unemployment_days = days_between(leave_date, join_date)
    daily_last_salary = args.last_salary / 365.0
    lost_earnings = round(daily_last_salary * unemployment_days)

    # === Compound recovery time (fixed) ===
    recovery = calculate_recovery_time_compound(
        lost_earnings=lost_earnings,
        new_salary=args.new_salary,
        last_salary=args.last_salary,
        join_date=join_date,
        old_increase=args.old_increase,
        new_increase=args.new_increase,
    )
    cover_time = convert_days_to_ymd(recovery["days_count"])

    # Print summary
    print("\\n=== Recovery Summary (Compound Raises) ===")
    print(f"Unemployment period: {convert_days_to_ymd(unemployment_days)} ({unemployment_days} days)")
    print(f"Lost earnings: ${lost_earnings:,.0f}")
    print(f"Time to break even: {cover_time}")
    print(f"Total recovery amount: ${recovery['total_amount']:,.0f}")

    # === Year projection (unchanged logic, but now day-accurate) ===
    old_earnings, new_earnings, diff = project_salary_for_year(
        leave_date, join_date, args.last_salary, args.new_salary,
        args.old_increase, args.new_increase, args.year, verbose=args.verbose
    )

    print(f"\\n=== Projection for {args.year} ===")
    print(f"Old job earnings: ${old_earnings:,.0f}")
    print(f"New job earnings: ${new_earnings:,.0f}")
    print(f"Difference: ${diff:,.0f}")

if __name__ == "__main__":
    main()
