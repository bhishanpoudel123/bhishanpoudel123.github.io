document.addEventListener('DOMContentLoaded', function() {
    document.getElementById("calculate").addEventListener("click", function(e) {
        e.preventDefault();
        calculateSingleRecovery();
    });
});

function daysBetween(startDate, endDate) {
    const start = new Date(startDate);
    const end = new Date(endDate);
    const timeDiff = end.getTime() - start.getTime();
    return Math.ceil(timeDiff / (1000 * 3600 * 24));
}

function convertDaysToYMD(days) {
    const years = Math.floor(days / 365);
    const remainingDaysAfterYears = days % 365;
    const months = Math.floor(remainingDaysAfterYears / 30);
    const remainingDays = remainingDaysAfterYears % 30;
    return `${years} Years ${months} Months ${remainingDays} Days`;
}

function calculateRecoveryTime(lostEarnings, newSalary, lastSalary, joinDate, oldAnnualIncrease, newAnnualIncrease) {
    let cumulativeDifference = 0;
    let daysCount = 0;
    let currentDate = new Date(joinDate);
    let currentOldSalary = lastSalary;
    let currentNewSalary = newSalary;
    
    const maxDays = 3650; // 10 years safety limit
    
    while (cumulativeDifference < lostEarnings && daysCount < maxDays) {
        // Calculate daily earnings difference
        const dailyOldSalary = currentOldSalary / 365;
        const dailyNewSalary = currentNewSalary / 365;
        const dailyDifference = dailyNewSalary - dailyOldSalary;
        
        cumulativeDifference += dailyDifference;
        daysCount++;
        
        // Move to next day
        currentDate.setDate(currentDate.getDate() + 1);
        
        // Check for October 1st salary increases
        if (currentDate.getMonth() === 9 && currentDate.getDate() === 1) {
            currentOldSalary *= (1 + oldAnnualIncrease / 100);
            currentNewSalary *= (1 + newAnnualIncrease / 100);
        }
    }
    
    return { 
        daysCount, 
        totalAmount: Math.round(cumulativeDifference),
        dailyOld: currentOldSalary / 365,
        dailyNew: currentNewSalary / 365,
        cumOld: (currentOldSalary / 365) * daysCount,
        cumNew: (currentNewSalary / 365) * daysCount,
        breakevenDate: currentDate
    };
}

function formatBreakevenInfo(recoveryInfo, leaveDate, joinDate) {
    const today = new Date();
    const breakevenDate = recoveryInfo.breakevenDate;
    
    // Calculate time differences
    function getTimeDiff(start, end) {
        const years = end.getFullYear() - start.getFullYear();
        let months = end.getMonth() - start.getMonth();
        let days = end.getDate() - start.getDate();
        
        if (days < 0) {
            months -= 1;
            days += new Date(end.getFullYear(), end.getMonth(), 0).getDate();
        }
        
        if (months < 0) {
            months += 12;
        }
        
        return `${years}y ${months}m ${days}d`;
    }
    
    const fromLeave = getTimeDiff(new Date(leaveDate), breakevenDate);
    const fromJoin = getTimeDiff(new Date(joinDate), breakevenDate);
    const fromToday = getTimeDiff(today, breakevenDate);
    
    return `
        <div class="breakeven-details">
            <h4>📅 Breakeven Date Analysis</h4>
            <p><strong>Status:</strong> ${recoveryInfo.totalAmount >= 0 ? 'Breakeven Reached' : 'Closest Calculation'}</p>
            <p><strong>Date:</strong> ${breakevenDate.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</p>
            <p><strong>Cumulative Old Job Earnings:</strong> $${Math.round(recoveryInfo.cumOld).toLocaleString()}</p>
            <p><strong>Cumulative New Job Earnings:</strong> $${Math.round(recoveryInfo.cumNew).toLocaleString()}</p>
            <p><strong>Gap (New - Old):</strong> $${recoveryInfo.totalAmount.toLocaleString()}</p>
            
            <div class="salary-details">
                <div>
                    <h5>Old Job</h5>
                    <p>Daily: $${recoveryInfo.dailyOld.toFixed(2)}</p>
                    <p>Annual: $${Math.round(recoveryInfo.dailyOld * 365).toLocaleString()}</p>
                </div>
                <div>
                    <h5>New Job</h5>
                    <p>Daily: $${recoveryInfo.dailyNew.toFixed(2)}</p>
                    <p>Annual: $${Math.round(recoveryInfo.dailyNew * 365).toLocaleString()}</p>
                </div>
            </div>
            
            <div class="time-details">
                <p><strong>Time from leave date:</strong> ${fromLeave}</p>
                <p><strong>Time from join date:</strong> ${fromJoin}</p>
                <p><strong>Time from today:</strong> ${fromToday}</p>
            </div>
        </div>
    `;
}

function calculateSingleRecovery() {
    try {
        // Get input values
        const leaveDate = document.getElementById("leave-date").value;
        const joinDate = document.getElementById("join-date").value;
        const lastSalary = parseFloat(document.getElementById("last-salary").value) || 0;
        const newSalary = parseFloat(document.getElementById("new-salary").value) || 0;
        const oldAnnualIncrease = parseFloat(document.getElementById("old-annual-increase").value) || 0;
        const newAnnualIncrease = parseFloat(document.getElementById("new-annual-increase").value) || 0;

        // Validation
        if (isNaN(new Date(leaveDate).getTime()) || isNaN(new Date(joinDate).getTime())) {
            throw new Error("Invalid dates provided");
        }
        if (lastSalary <= 0 || newSalary <= 0) {
            throw new Error("Salaries must be positive numbers");
        }
        if (new Date(leaveDate) >= new Date(joinDate)) {
            throw new Error("Join date must be after leave date");
        }
        if (newSalary <= lastSalary) {
            throw new Error("New salary must be higher than last salary");
        }

        // Calculate basic values
        const unemploymentDays = daysBetween(leaveDate, joinDate);
        const dailyLastSalary = lastSalary / 365;
        const lostEarnings = Math.round(dailyLastSalary * unemploymentDays);

        // Calculate recovery time
        const recovery = calculateRecoveryTime(
            lostEarnings, 
            newSalary, 
            lastSalary, 
            new Date(joinDate), 
            oldAnnualIncrease, 
            newAnnualIncrease
        );
        
        const coverTime = convertDaysToYMD(recovery.daysCount);
        const breakevenInfo = formatBreakevenInfo(recovery, leaveDate, joinDate);

        // Display results
        document.getElementById("result").innerHTML = `
            <div class="summary">
                <h2>📊 Job Loss Recovery Analysis</h2>
                
                <div class="section unemployment-section">
                    <h3>📉 Unemployment Period</h3>
                    <p><strong>Duration:</strong> ${convertDaysToYMD(unemploymentDays)} (${unemploymentDays} days)</p>
                    <p><strong>Lost Earnings:</strong> $${lostEarnings.toLocaleString()}</p>
                    <p><strong>Daily Loss:</strong> $${dailyLastSalary.toFixed(2)}</p>
                </div>
                
                <div class="comparison">
                    <div class="job-box old-job">
                        <h3>🔴 Previous Job</h3>
                        <p><strong>Salary:</strong> $${lastSalary.toLocaleString()}/year</p>
                        <p><strong>Daily Rate:</strong> $${dailyLastSalary.toFixed(2)}</p>
                        <p><strong>Annual Increase:</strong> ${oldAnnualIncrease}%</p>
                    </div>
                    
                    <div class="job-box new-job">
                        <h3>🟢 New Job</h3>
                        <p><strong>Salary:</strong> $${newSalary.toLocaleString()}/year</p>
                        <p><strong>Daily Rate:</strong> $${(newSalary / 365).toFixed(2)}</p>
                        <p><strong>Daily Advantage:</strong> $${((newSalary / 365) - dailyLastSalary).toFixed(2)}</p>
                        <p><strong>Annual Increase:</strong> ${newAnnualIncrease}%</p>
                    </div>
                </div>
                
                <div class="recovery-section">
                    <h3>🎯 Recovery Timeline</h3>
                    <p><strong>Time to Break Even:</strong> ${coverTime}</p>
                    <p><strong>Total Recovery Amount:</strong> $${recovery.totalAmount.toLocaleString()}</p>
                    ${breakevenInfo}
                </div>
            </div>
        `;

    } catch (error) {
        document.getElementById("result").innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${error.message}
            </div>
        `;
    }
}