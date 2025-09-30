# Introduction
- [Main Website](https://bhishanpoudel123.github.io)
- [Github Repo](https://github.com/bhishanpoudel123/bhishanpoudel123.github.io)

# Webapps
- [Job Calculator ](https://bhishanpoudel123.github.io/job)
- [Stock Calculator](https://bhishanpoudel123.github.io/stock)
- [Stock Calculator Buy](https://bhishanpoudel123.github.io/stock/buy)
- [Stock Calculator Recovery](https://bhishanpoudel123.github.io/stock/recovery)

# 📊 Stock Calculator Suite

A comprehensive web-based tool for stock investment calculations, featuring multiple calculators to help you make informed decisions about averaging down and recovery strategies.

## 🚀 Features

### 1. 📈 Stock Buy Calculator For Loss Status

**Purpose**: Calculate how many additional shares you need to buy to achieve a target average price when you're in a loss position.

**When to Use**: 
- Your current stock buying average is **ABOVE** the current market price (you're at a loss)
- You want to buy more shares at the lower current price to reduce your average cost
- You're implementing an "averaging down" strategy

**Key Metrics Calculated**:
- **Shares to Buy**: Number of additional shares needed at current price to reach target average
- **Total Cost for New Purchase**: Investment amount required for the new purchase
- **Current Average Price**: Your existing average cost per share before buying more
- **Price Increase Needed for Break-Even**: Percentage the stock must rise from current price to reach your current average

**Default Example Values**:
- Robinhood: 3,500 shares at $5.72
- Webull: 3,100 shares at $6.60
- Current Stock Price: $4.00
- Target Average Price: $6.00

**How It Works**:
```
Formula: (Total Cost + New Shares × Current Price) / (Total Shares + New Shares) = Target Average
```

The calculator solves for "New Shares" to determine how many you need to buy.

**Example Scenario**:
- You bought 6,600 shares with an average cost of $6.14
- Stock dropped to $4.00
- You want to lower your average to $6.00
- Calculator shows you need to buy X shares at $4.00 to achieve this

---

### 2. 📉 Stock Recovery Calculator

**Purpose**: Calculate what stock price you need to reach to recover from a realized loss using your current holdings.

**When to Use**:
- You've already sold stocks at a loss (realized loss)
- You currently own shares in the same or different stock
- You want to know what price your current holdings need to reach to offset previous losses

**Key Metrics Calculated**:
- **Current Portfolio Value**: What your current shares are worth now
- **Target Portfolio Value**: Total value needed to recover the realized loss
- **Required Stock Price for Recovery**: Price per share needed to break even
- **Price Increase Needed**: Dollar amount and percentage increase required

**Default Example Values**:
- Realized Loss: $41,000
- Current Stock Price: $7.55
- Number of Shares Owned: 3,500

**How It Works**:
```
Target Value = Current Portfolio Value + Realized Loss
Recovery Price = Target Value / Number of Shares
```

**Example Scenario**:
- You previously lost $41,000 on stock sales
- You now own 3,500 shares at $7.55 each
- Current portfolio value: $26,425
- Calculator shows the stock needs to reach $19.26 to fully recover your loss

---

## 💡 Key Differences Between Calculators

| Feature | Buy Calculator (Loss Status) | Recovery Calculator |
|---------|------------------------------|---------------------|
| **Purpose** | Lower average cost by buying more | Calculate break-even price after realized loss |
| **Scenario** | Currently holding losing positions | Already sold at a loss, holding new position |
| **Calculates** | How many shares to buy | What price to reach |
| **Strategy** | Averaging down | Recovery planning |
| **Loss Type** | Unrealized loss | Realized loss |

---

## 🎯 When to Use Each Calculator

### Use Buy Calculator When:
- ✅ You're holding stocks currently at a loss (unrealized)
- ✅ You believe in the stock's long-term potential
- ✅ You have additional capital to invest
- ✅ You want to strategically lower your average cost
- ✅ Current price is below your buying average

### Use Recovery Calculator When:
- ✅ You've already closed positions at a loss
- ✅ You want to understand recovery targets
- ✅ Planning future exit strategies
- ✅ Tracking overall portfolio recovery goals
- ✅ Need to know break-even price for current holdings

---

## 📱 Technical Features

- **Fully Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Mobile-Friendly Interface**: 
  - Collapsible sidebar navigation on mobile
  - Touch-optimized buttons and inputs
  - Responsive grid layouts
- **Real-time Calculations**: Instant results as you input data
- **Dynamic Input Management**: Add/remove multiple purchase entries
- **Input Validation**: Prevents calculation errors with helpful alerts
- **Auto-calculation**: Results display automatically with default values
- **Modern UI/UX**: Clean, intuitive interface with gradient design

---

## 🛠️ Technology Stack

- **HTML5**: Semantic markup
- **CSS3**: Modern styling with Flexbox and Grid
- **Vanilla JavaScript**: No dependencies, lightweight and fast
- **Responsive Design**: Media queries for all screen sizes

---

## 📖 How to Use

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/stock-calculator-suite.git
   ```

2. **Open in browser**
   - Simply open `index.html` in your web browser
   - No build process or dependencies required

3. **Navigate between calculators**
   - Use the sidebar menu to switch between different calculators
   - On mobile, tap the hamburger menu (☰) to access navigation

---

## ⚠️ Important Disclaimers

- **Not Financial Advice**: These calculators are tools for mathematical computation only
- **Use at Your Own Risk**: Always do your own research before making investment decisions
- **Averaging Down Risks**: Buying more of a losing stock increases your exposure and risk
- **Market Volatility**: Stock prices can continue to fall beyond your projections
- **Consult Professionals**: Consider consulting with a financial advisor for personalized advice

---

## 🎓 Investment Strategy Notes

### Averaging Down Strategy
**Pros:**
- Lowers your average cost per share
- Potentially faster break-even when stock recovers
- Accumulate more shares at discount prices

**Cons:**
- Increases total capital at risk
- Stock may continue declining
- Opportunity cost of capital
- "Catching a falling knife" risk

**Best Practices:**
- Only average down on fundamentally strong companies
- Have a predetermined limit (don't keep averaging down indefinitely)
- Ensure adequate diversification
- Don't invest money you can't afford to lose

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

Created for personal portfolio tracking and investment planning.

---

## 📞 Support

If you have questions or need help:
- Open an issue on GitHub
- Check the inline help text within each calculator

---

## 🔄 Version History

- **v1.0.0** (2025-09-30)
  - Initial release
  - Buy Calculator for Loss Status
  - Recovery Calculator
  - Mobile-responsive design
  - Default value support

---

## 🎯 Future Enhancements

- [ ] Export calculations to PDF
- [ ] Save calculation history
- [ ] Multiple portfolio tracking
- [ ] Chart visualizations
- [ ] Tax loss harvesting calculator
- [ ] Dividend reinvestment calculator
- [ ] Options strategy calculator

---

**Remember**: Past performance does not guarantee future results. Invest wisely! 📈
