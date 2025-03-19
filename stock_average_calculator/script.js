document.getElementById("add").addEventListener("click", function() {
    let inputGroup = document.createElement("div");
    inputGroup.classList.add("input-group");
    inputGroup.innerHTML = `
        <input type="number" placeholder="Price per Share ($)" class="price">
        <input type="number" placeholder="Number of Shares" class="shares">
        <button class="remove">✖</button>
    `;
    document.getElementById("inputs").appendChild(inputGroup);

    inputGroup.querySelector(".remove").addEventListener("click", function() {
        inputGroup.remove();
    });
});

document.getElementById("calculate").addEventListener("click", function() {
    let prices = document.querySelectorAll(".price");
    let shares = document.querySelectorAll(".shares");
    let currentPrice = parseFloat(document.getElementById("current-price").value);
    let targetPrice = parseFloat(document.getElementById("target-price").value);
    
    let totalCost = 0;
    let totalShares = 0;

    for (let i = 0; i < prices.length; i++) {
        let price = parseFloat(prices[i].value);
        let numShares = parseFloat(shares[i].value);

        if (!isNaN(price) && !isNaN(numShares)) {
            totalCost += price * numShares;
            totalShares += numShares;
        }
    }

    // Validate inputs
    if (totalShares === 0 || isNaN(currentPrice) || isNaN(targetPrice) || currentPrice <= 0 || targetPrice <= 0) {
        document.getElementById("result").textContent = "Enter valid values!";
        return;
    }

    let avgCost = totalCost / totalShares; // Current average cost

    // Number of additional shares required
    let requiredShares = Math.ceil((totalCost - (targetPrice * totalShares)) / (targetPrice - currentPrice));
    let requiredAmount = requiredShares * currentPrice; // Total cost of buying those shares

    // Format requiredAmount as an integer with thousands separator
    let formattedRequiredAmount = Math.round(requiredAmount).toLocaleString();

    // Percentage increase needed for current price to match average cost
    let increaseToAvg = ((avgCost - currentPrice) / currentPrice) * 100;

    // Percentage increase needed for current price to reach target price
    let increaseToTarget = ((targetPrice - currentPrice) / currentPrice) * 100;

    // Display results
    let resultHTML = `
        You need to buy: <strong>${requiredShares} shares</strong><br>
        Total Cost: <strong>$${formattedRequiredAmount}</strong><br><br>
        <strong>Price Increase Needed:</strong><br>
        Match average cost ($${avgCost.toFixed(2)}): <strong>${increaseToAvg.toFixed(2)}%</strong><br>
        Match target price ($${targetPrice.toFixed(2)}): <strong>${increaseToTarget.toFixed(2)}%</strong>
    `;

    document.getElementById("result").innerHTML = resultHTML;
});

