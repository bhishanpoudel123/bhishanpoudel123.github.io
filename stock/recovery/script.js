document.getElementById("calculate").addEventListener("click", function() {
    let realisedLoss = parseFloat(document.getElementById("realised-loss").value);
    let currentPrice = parseFloat(document.getElementById("current-price").value);
    let numShares = parseFloat(document.getElementById("num-shares").value);

    // Validate inputs
    if (isNaN(realisedLoss) || isNaN(currentPrice) || isNaN(numShares) || numShares <= 0 || currentPrice <= 0) {
        document.getElementById("result").innerHTML = "⚠️ Please enter valid values!";
        return;
    }

    // Calculate required price to recover losses
    let requiredPrice = (realisedLoss / numShares) + currentPrice;

    // Display result with formatted number
    document.getElementById("result").innerHTML = `📢 Required price: <strong>$${requiredPrice.toFixed(2)}</strong>`;
});

// Trigger calculation on Enter key press in target-price input
document.getElementById("num-shares").addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
        event.preventDefault();
        document.getElementById("calculate").click();
    }
});