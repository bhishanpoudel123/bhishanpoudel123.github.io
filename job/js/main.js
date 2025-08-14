document.addEventListener('DOMContentLoaded', function() {
    document.getElementById("calculate").addEventListener("click", calculateSingleRecovery);
    document.getElementById("calculate-range").addEventListener("click", calculateRangeAnalysis);
});

function calculateSingleRecovery() {
    try {
        let leaveDate = new Date(document.getElementById("leave-date").value);
        let joinDate = new Date(document.getElementById("join-date").value);
        let lastSalary = parseFloat(document.getElementById("last-salary").value) || 111000;
        let newSalary = parseFloat(document.getElementById("new-salary").value) || 130000;
        let oldAnnualIncrease = parseFloat(document.getElementById("old-annual-increase").value) || 3;
        let newAnnualIncrease = parseFloat(document.getElementById("new-annual-increase").value) || 0;

        // Validation
        if (isNaN(lastSalary) || isNaN(newSalary) || leaveDate >= joinDate || newSalary <= lastSalary) {
            throw new Error("Invalid input values");
        }

        let unemploymentDays = daysBetween(leaveDate, joinDate);
        let dailyLastSalary = lastSalary / 365;
        let dailyNewSalary = newSalary / 365;
        let lostEarnings = Math.round(dailyLastSalary * unemploymentDays);
        let dailyDifference = dailyNewSalary - dailyLastSalary;
        let requiredDays = Math.ceil(lostEarnings / dailyDifference);
        let totalAmount = Math.round(requiredDays * dailyNewSalary);
        let coverTime = convertDaysToYMD(requiredDays);

        document.getElementById("result").innerHTML = `
            <div class="summary">
                <h3>💼 Recovery Analysis Summary</h3>
                <div class="comparison">
                    <div class="job-box laid-off">
                        <h4>📉 Unemployment Period</h4>
                        <p><strong>Duration:</strong> ${convertDaysToYMD(unemploymentDays)}</p>
                        <p><strong>Lost Earnings:</strong> $${lostEarnings.toLocaleString()}</p>
                        <p><strong>Daily Loss:</strong> $${dailyLastSalary.toFixed(2)}/day</p>
                    </div>
                    
                    <div class="job-box old-job">
                        <h4>🔴 Previous Job</h4>
                        <p><strong>Salary:</strong> $${lastSalary.toLocaleString()}/year</p>
                        <p><strong>Daily Rate:</strong> $${dailyLastSalary.toFixed(2)}/day</p>
                        <p><strong>Annual Increase:</strong> ${oldAnnualIncrease}%</p>
                    </div>
                    
                    <div class="job-box new-job">
                        <h4>🟢 New Job</h4>
                        <p><strong>Salary:</strong> $${newSalary.toLocaleString()}/year</p>
                        <p><strong>Daily Rate:</strong> $${dailyNewSalary.toFixed(2)}/day</p>
                        <p><strong>Daily Advantage:</strong> $${dailyDifference.toFixed(2)}/day</p>
                        <p><strong>Annual Increase:</strong> ${newAnnualIncrease}%</p>
                    </div>
                </div>
                
                <div style="background: #e3f2fd; padding: 20px; border-radius: 8px; margin-top: 20px; text-align: center;">
                    <h4 style="color: #1976d2; margin-bottom: 10px;">🎯 Recovery Timeline</h4>
                    <p style="font-size: 18px; margin: 5px 0;"><strong>Time to Break Even:</strong> ${coverTime}</p>
                    <p style="font-size: 16px; margin: 5px 0;"><strong>Total Recovery Amount:</strong> $${totalAmount.toLocaleString()}</p>
                    <p style="font-size: 14px; color: #666;">You'll recover your unemployment losses after earning this much in your new job</p>
                </div>
            </div>
        `;

    } catch (error) {
        console.error("Calculation error:", error);
        document.getElementById("result").innerHTML = `
            <div class="error-message">
                Error in calculation: ${error.message}
            </div>
        `;
    }
}

function calculateRangeAnalysis() {
    try {
        let leaveDate = new Date(document.getElementById("range-leave-date").value);
        let joinDate = new Date(document.getElementById("range-join-date").value);
        let lastSalary = parseFloat(document.getElementById("range-last-salary").value) || 111000;
        let oldAnnualIncrease = parseFloat(document.getElementById("range-old-annual-increase").value) || 3;
        let minSalary = parseFloat(document.getElementById("range-min-salary").value) || 120000;
        let maxSalary = parseFloat(document.getElementById("range-max-salary").value) || 200000;
        let newAnnualIncrease = parseFloat(document.getElementById("range-new-annual-increase").value) || 0;

        // Validation
        if (isNaN(lastSalary) || isNaN(minSalary) || isNaN(maxSalary) || 
            leaveDate >= joinDate || minSalary >= maxSalary || minSalary <= lastSalary) {
            throw new Error("Invalid input values");
        }

        let unemploymentDays = daysBetween(leaveDate, joinDate);
        let dailyLastSalary = lastSalary / 365;
        let lostEarnings = Math.round(dailyLastSalary * unemploymentDays);

        // Generate table
        let baseYear = joinDate.getFullYear();
        let yearHeaders = '';
        for (let i = 0; i < 10; i++) {
            yearHeaders += `<th>${baseYear + i}</th>`;
        }

        // Pre-calculate old job earnings for all years (this should be the same for all rows)
        let oldJobEarnings = [];
        for (let yearOffset = 0; yearOffset < 10; yearOffset++) {
            const targetYear = baseYear + yearOffset;
            const endDate = new Date(targetYear, leaveDate.getMonth(), leaveDate.getDate());
            const oldEarnings = calculateAnnualEarnings(leaveDate, endDate, lastSalary, oldAnnualIncrease);
            oldJobEarnings.push(oldEarnings);
            console.log(`Pre-calculated old job earnings for ${targetYear}: ${oldEarnings}`);
        }

        let tableRows = '';
        const increment = 5000;

        for (let currentSalary = minSalary; currentSalary <= maxSalary; currentSalary += increment) {
            let dailyNewSalary = currentSalary / 365;
            let requiredDays = Math.ceil(lostEarnings / (dailyNewSalary - dailyLastSalary));
            let totalAmount = Math.round(requiredDays * dailyNewSalary);
            let coverTime = convertDaysToYMD(requiredDays);
            let salaryDifference = currentSalary - lastSalary;

            let futureSalaries = '';
            
            for (let yearOffset = 0; yearOffset < 10; yearOffset++) {
                const targetYear = baseYear + yearOffset;
                const endDate = new Date(targetYear, leaveDate.getMonth(), leaveDate.getDate());
                
                // Use pre-calculated old job earnings (same for all rows)
                let oldEarnings = oldJobEarnings[yearOffset];
                // Calculate new job earnings for this specific salary
                let newEarnings = endDate > joinDate ? 
                    calculateAnnualEarnings(joinDate, endDate, currentSalary, newAnnualIncrease) : 0;
                let difference = newEarnings - oldEarnings;

                console.log(`Row ${currentSalary}: Year ${targetYear} - Old: ${oldEarnings}, New: ${newEarnings}, Diff: ${difference}`);

                futureSalaries += `
                    <td>
                        <div class="salary-stack">
                            <div class="old-salary">${oldEarnings.toLocaleString()}</div>
                            <div class="new-salary">${newEarnings.toLocaleString()}</div>
                            <div class="difference-value">${difference.toLocaleString()}</div>
                        </div>
                    </td>
                `;
            }

            tableRows += `
                <tr>
                    <td>
                        <div class="salary-stack">
                            <div class="old-salary">${lastSalary.toLocaleString()}</div>
                            <div class="new-salary">${currentSalary.toLocaleString()}</div>
                            <div class="difference-value">${salaryDifference.toLocaleString()}</div>
                        </div>
                    </td>
                    <td>${coverTime}</td>
                    <td>${totalAmount.toLocaleString()}</td>
                    ${futureSalaries}
                </tr>
            `;
        }

        document.getElementById("range-result").innerHTML = `
            <div class="summary">
                <h3>📊 Range Analysis Summary</h3>
                <p><strong>Unemployment Period:</strong> ${convertDaysToYMD(unemploymentDays)} (${unemploymentDays} days)</p>
                <p><strong>Lost Earnings:</strong> $${lostEarnings.toLocaleString()}</p>
                <p><strong>Old Job:</strong> $${lastSalary.toLocaleString()}/year with ${oldAnnualIncrease}% annual increases</p>
                <p><strong>New Job Range:</strong> $${minSalary.toLocaleString()} - $${maxSalary.toLocaleString()}/year with ${newAnnualIncrease}% annual increases</p>
                <p style="font-size: 12px; color: #666; margin-top: 15px;">
                    Each year column shows: <span class="old-salary">Old Job Earnings</span> | 
                    <span class="new-salary">New Job Earnings</span> | 
                    <span class="difference-value">Difference</span>
                </p>
            </div>
            <div class="table-container">
                <table class="analysis-table">
                    <thead>
                        <tr>
                            <th>💰 Base Salary</th>
                            <th>⏱️ Cover Time</th>
                            <th>💸 Recovery Amount</th>
                            ${yearHeaders}
                        </tr>
                    </thead>
                    <tbody>
                        ${tableRows}
                    </tbody>
                </table>
            </div>
        `;

    } catch (error) {
        console.error("Calculation error:", error);
        document.getElementById("range-result").innerHTML = `
            <div class="error-message">
                Error in calculation: ${error.message}
            </div>
        `;
    }
}

document.getElementById("info-button").addEventListener("click", showCalculationInfo);

function showCalculationInfo() {
    try {
        const leaveDate = new Date(document.getElementById("range-leave-date").value);
        const joinDate = new Date(document.getElementById("range-join-date").value);
        const lastSalary = parseFloat(document.getElementById("range-last-salary").value) || 111000;
        const oldAnnualIncrease = parseFloat(document.getElementById("range-old-annual-increase").value) || 3;
        const newSalary = parseFloat(document.getElementById("range-min-salary").value) || 120000;
        const newAnnualIncrease = parseFloat(document.getElementById("range-new-annual-increase").value) || 0;

        // Example calculation for old job (first year)
        const oldEndDate = new Date(leaveDate);
        oldEndDate.setFullYear(leaveDate.getFullYear() + 1);
        const oldEarnings = calculateAnnualEarnings(leaveDate, oldEndDate, lastSalary, oldAnnualIncrease);

        // Example calculation for new job (first year)
        const newEndDate = new Date(joinDate);
        newEndDate.setFullYear(joinDate.getFullYear() + 1);
        const newEarnings = calculateAnnualEarnings(joinDate, newEndDate, newSalary, newAnnualIncrease);

        const infoDiv = document.getElementById("info-content");
        infoDiv.style.display = infoDiv.style.display === "none" ? "block" : "none";
        
        infoDiv.innerHTML = `
            <div class="info-content">
                <h3>📝 Salary Calculation Explanation</h3>
                
                <div class="info-step">
                    <h4>1. Unemployment Period</h4>
                    <p>From <strong>${leaveDate.toDateString()}</strong> to <strong>${joinDate.toDateString()}</strong></p>
                    <p>Duration: <span class="highlight">${daysBetween(leaveDate, joinDate)} days</span></p>
                    <p>Lost earnings: <span class="highlight">$${Math.round((lastSalary / 365) * daysBetween(leaveDate, joinDate)).toLocaleString()}</span></p>
                </div>
                
                <div class="info-step">
                    <h4>2. Old Job Salary Calculation (First Year)</h4>
                    <p>Base salary: <span class="highlight">$${lastSalary.toLocaleString()}</span></p>
                    <p>Annual increase: <span class="highlight">${oldAnnualIncrease}%</span></p>
                    <p>Period: <span class="highlight">${leaveDate.toDateString()} to ${oldEndDate.toDateString()}</span></p>
                    <p>Earnings: <span class="highlight">$${oldEarnings.toLocaleString()}</span></p>
                    <p>Calculation method: Daily rate applied to each day between October salary adjustments</p>
                </div>
                
                <div class="info-step">
                    <h4>3. New Job Salary Calculation (First Year)</h4>
                    <p>Base salary: <span class="highlight">$${newSalary.toLocaleString()}</span></p>
                    <p>Annual increase: <span class="highlight">${newAnnualIncrease}%</span></p>
                    <p>Period: <span class="highlight">${joinDate.toDateString()} to ${newEndDate.toDateString()}</span></p>
                    <p>Earnings: <span class="highlight">$${newEarnings.toLocaleString()}</span></p>
                    <p>Calculation method: Daily rate applied to each day between October salary adjustments</p>
                </div>
                
                <div class="info-step">
                    <h4>4. Recovery Calculation</h4>
                    <p>Daily advantage: <span class="highlight">$${(newSalary/365 - lastSalary/365).toFixed(2)}/day</span></p>
                    <p>Days to recover: <span class="highlight">${Math.ceil((lastSalary/365 * daysBetween(leaveDate, joinDate)) / (newSalary/365 - lastSalary/365))} days</span></p>
                    <p>Total recovery amount: <span class="highlight">$${Math.ceil((lastSalary/365 * daysBetween(leaveDate, joinDate)) / (newSalary/365 - lastSalary/365) * newSalary/365).toLocaleString()}</span></p>
                </div>
                
                <div class="info-step">
                    <h4>Key Points</h4>
                    <p>• Salaries are calculated on a daily basis (annual salary ÷ 365)</p>
                    <p>• Salary increases happen every October 1st</p>
                    <p>• Recovery time depends on the difference between new and old daily salaries</p>
                    <p>• The table shows projections for 10 years with annual comparisons</p>
                </div>
            </div>
        `;

    } catch (error) {
        console.error("Info error:", error);
        document.getElementById("info-content").innerHTML = `
            <div class="error-message">
                Error showing info: ${error.message}
            </div>
        `;
    }
}