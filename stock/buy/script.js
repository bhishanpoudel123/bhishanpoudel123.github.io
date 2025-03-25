document.getElementById("calculate").addEventListener("click", function () {
    let price = parseFloat(document.getElementById("price").value);
    let numShares = parseFloat(document.getElementById("shares").value);
    let currentPrice = parseFloat(document.getElementById("current-price").value);
    let targetPrice = parseFloat(document.getElementById("target-price").value);

    if (isNaN(price) || isNaN(numShares) || isNaN(currentPrice) || isNaN(targetPrice) || price <= 0 || numShares <= 0 || currentPrice <= 0 || targetPrice <= 0) {
        document.getElementById("result").textContent = "Enter valid values!";
        return;
    }

    let totalCost = price * numShares;
    let requiredShares = Math.ceil((totalCost - (targetPrice * numShares)) / (targetPrice - currentPrice));
    let requiredAmount = requiredShares * currentPrice;

    let formattedRequiredAmount = Math.round(requiredAmount).toLocaleString();
    let avgCost = totalCost / numShares;
    let increaseToAvg = ((avgCost - currentPrice) / currentPrice) * 100;
    let increaseToTarget = ((targetPrice - currentPrice) / currentPrice) * 100;

    let resultHTML = `
        You need to buy: <strong>${requiredShares} shares</strong><br>
        Total Cost: <strong>$${formattedRequiredAmount}</strong><br><br>
        <strong>Price Increase Needed:</strong><br>
        Match average cost ($${avgCost.toFixed(2)}): <strong>${increaseToAvg.toFixed(2)}%</strong><br>
        Match target price ($${targetPrice.toFixed(2)}): <strong>${increaseToTarget.toFixed(2)}%</strong>
    `;

    document.getElementById("result").innerHTML = resultHTML;
});

// Trigger calculation on Enter key press in target-price input
document.getElementById("target-price").addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
        event.preventDefault();
        document.getElementById("calculate").click();
    }
});
