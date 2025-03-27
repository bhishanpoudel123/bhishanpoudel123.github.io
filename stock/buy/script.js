document.getElementById("add").addEventListener("click", function () {
    let inputContainer = document.getElementById("inputs");
    let newGroup = document.createElement("div");
    newGroup.classList.add("input-group");
    newGroup.innerHTML = `
        <input type="text" placeholder="Vendor" class="vendor">
        <input type="number" placeholder="Shares" class="shares">
        <input type="number" placeholder="Price per Share ($)" class="price">
        <button class="remove">X</button>
    `;
    inputContainer.appendChild(newGroup);

    newGroup.querySelector(".remove").addEventListener("click", function () {
        inputContainer.removeChild(newGroup);
    });
});

document.getElementById("calculate").addEventListener("click", function () {
    let currentPrice = parseFloat(document.getElementById("current-price").value);
    let targetPrice = parseFloat(document.getElementById("target-price").value);
    let totalShares = 0, totalCost = 0;

    document.querySelectorAll(".input-group").forEach(group => {
        let shares = parseFloat(group.querySelector(".shares").value) || 0;
        let price = parseFloat(group.querySelector(".price").value) || 0;
        totalShares += shares;
        totalCost += shares * price;
    });

    if (totalShares === 0 || isNaN(currentPrice) || isNaN(targetPrice) || currentPrice <= 0 || targetPrice <= 0) {
        document.getElementById("result").textContent = "Enter valid values!";
        return;
    }

    let avgCost = totalCost / totalShares;
    let requiredShares = Math.ceil((totalCost - (targetPrice * totalShares)) / (targetPrice - currentPrice));
    let requiredAmount = requiredShares * currentPrice;
    let formattedRequiredAmount = Math.round(requiredAmount).toLocaleString();

    document.getElementById("result").innerHTML = `
        You need to buy: <strong>${requiredShares} shares</strong><br>
        Total Cost: <strong>$${formattedRequiredAmount}</strong>
    `;
});
