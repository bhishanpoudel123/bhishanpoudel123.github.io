// Helper function to calculate days between dates
function daysBetween(startDate, endDate) {
    // Make sure we're calculating from the start of start date to start of end date
    const start = new Date(startDate.getFullYear(), startDate.getMonth(), startDate.getDate());
    const end = new Date(endDate.getFullYear(), endDate.getMonth(), endDate.getDate());
    return Math.round((end - start) / (1000 * 60 * 60 * 24));
}

// Helper function to convert days to YMD format
function convertDaysToYMD(days) {
    const start = new Date(2000, 0, 1);
    const end = new Date(start);
    end.setDate(start.getDate() + days);

    let years = end.getFullYear() - start.getFullYear();
    let months = end.getMonth() - start.getMonth();
    let extraDays = end.getDate() - start.getDate();

    if (extraDays < 0) {
        months -= 1;
        extraDays += new Date(end.getFullYear(), end.getMonth(), 0).getDate();
    }

    if (months < 0) {
        years -= 1;
        months += 12;
    }

    return `${years}y ${months}m ${extraDays}d`;
}

// Fixed function to calculate annual earnings with proper date handling
function calculateAnnualEarnings(startDate, endDate, baseSalary, annualIncrease) {
    if (startDate >= endDate) return 0;
    
    console.log(`Calculating from ${startDate.toDateString()} to ${endDate.toDateString()}`);
    console.log(`Base salary: ${baseSalary}, Increase: ${annualIncrease}%`);
    
    let totalEarnings = 0;
    let currentDate = new Date(startDate.getFullYear(), startDate.getMonth(), startDate.getDate());
    let currentSalary = baseSalary;
    let periodCount = 1;
    
    while (currentDate < endDate) {
        // Find next October 1st or end date
        let nextOct1 = new Date(currentDate.getFullYear(), 9, 1); // October 1st
        if (currentDate >= nextOct1) {
            nextOct1 = new Date(currentDate.getFullYear() + 1, 9, 1);
        }
        
        let periodEnd = new Date(Math.min(nextOct1.getTime(), endDate.getTime()));
        let daysInPeriod = daysBetween(currentDate, periodEnd);
        let dailySalary = currentSalary / 365;
        let periodEarnings = dailySalary * daysInPeriod;
        
        console.log(`Period ${periodCount}: ${currentDate.toDateString()} to ${periodEnd.toDateString()}`);
        console.log(`Days: ${daysInPeriod}, Daily salary: ${dailySalary.toFixed(2)}, Earnings: ${periodEarnings.toFixed(2)}`);
        
        totalEarnings += periodEarnings;
        currentDate = new Date(periodEnd);
        
        // Apply salary increase if we hit October 1st and there's more time
        if (currentDate < endDate && currentDate.getTime() === nextOct1.getTime()) {
            currentSalary *= (1 + annualIncrease / 100);
            console.log(`New salary after ${annualIncrease}% increase: ${currentSalary}`);
        }
        periodCount++;
    }
    
    console.log(`Total earnings: ${totalEarnings.toFixed(2)}, Rounded: ${Math.round(totalEarnings)}`);
    return Math.round(totalEarnings);
}