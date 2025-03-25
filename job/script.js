document.getElementById("calculate").addEventListener("click", function () {
	let leaveDate = new Date(document.getElementById("leave-date").value);
	let joinDate = new Date(document.getElementById("join-date").value);
	let lastSalary = parseFloat(document.getElementById("last-salary").value) || 111000;
	let newSalary = parseFloat(document.getElementById("new-salary").value) || 140000;

	if (isNaN(lastSalary) || isNaN(newSalary) || leaveDate >= joinDate) {
		document.getElementById("result").textContent = "Enter valid values!";
		return;
	}

	let unemploymentDays = Math.round((joinDate - leaveDate) / (1000 * 60 * 60 * 24));
	let dailyLastSalary = lastSalary / 365;
	let lostEarnings = Math.round(dailyLastSalary * unemploymentDays);

	let dailyNewSalary = newSalary / 365;
	if (dailyNewSalary <= dailyLastSalary) {
		document.getElementById("result").textContent = "New salary must be greater than the last salary.";
		return;
	}

	let requiredDays = Math.ceil(lostEarnings / (dailyNewSalary - dailyLastSalary));
	let finalDate = new Date(joinDate.getTime() + requiredDays * 24 * 60 * 60 * 1000);

	// Adjust for proper date display
	let formattedLeaveDate = leaveDate.toLocaleDateString("en-US", { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' });
	let formattedJoinDate = joinDate.toLocaleDateString("en-US", { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' });
	let formattedFinalDate = finalDate.toLocaleDateString("en-US", { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' });

	// Earnings if stayed in old job
	let totalDaysIfStayed = Math.ceil((finalDate - leaveDate) / (1000 * 60 * 60 * 24));
	let earningsIfStayed = Math.round(totalDaysIfStayed * dailyLastSalary).toLocaleString();

	// Earnings in new job
	let earningsInNewJob = Math.round(requiredDays * dailyNewSalary).toLocaleString();

	// Display results with improved formatting
	document.getElementById("result").innerHTML = `
        <div class="summary">
            <h2>Recovery Summary</h2>
            <p><strong># Days:</strong> ${requiredDays}</p>
            <p><strong>Expected Recovery Date:</strong> ${formattedFinalDate}</p>
        </div>
        <div class="comparison">
            <div class="job-box laid-off">
                <h3>Laid Off Period</h3>
                <p><strong>Days Laid Off:</strong> ${unemploymentDays}</p>
                <p><strong>Daily Amount:</strong> $${Math.round(dailyLastSalary).toLocaleString()}</p>
                <p><strong>Total Lost Earnings:</strong> $${lostEarnings.toLocaleString()}</p>
            </div>
            <div class="job-box old-job">
                <h3>Old Job</h3>
                <p><strong>From:</strong> ${formattedLeaveDate}</p>
                <p><strong>To:</strong> ${formattedFinalDate}</p>
                <p><strong>Amount:</strong> $${earningsIfStayed}</p>
            </div>
            <div class="job-box new-job">
                <h3>New Job</h3>
                <p><strong>From:</strong> ${formattedJoinDate}</p>
                <p><strong>To:</strong> ${formattedFinalDate}</p>
                <p><strong>Amount:</strong> $${earningsInNewJob}</p>
            </div>
        </div>
    `;
});
