# Timeseries Quiz

## Table of Contents
- [Qn 01: What does the 'AR' component in ARIMA represent, and how does it capture patterns in time series data?](#1)
- [Qn 02: What does the 'I' component in ARIMA represent, and why is it necessary?](#2)
- [Qn 03: What does the 'MA' component in ARIMA represent, and how does it differ from AR?](#3)
- [Qn 04: How do you interpret the parameters p, d, and q in ARIMA(p,d,q)?](#4)
- [Qn 05: What is the key assumption that must be satisfied before applying ARIMA models?](#5)
- [Qn 06: How can you determine the appropriate values for p and q in an ARIMA(p,d,q) model?](#6)
- [Qn 07: What is the purpose of the Augmented Dickey-Fuller (ADF) test in time series analysis?](#7)
- [Qn 08: What is the difference between ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function)?](#8)
- [Qn 09: What additional component does SARIMAX add compared to ARIMA?](#9)
- [Qn 10: How are seasonality parameters represented in a SARIMA model?](#10)
- [Qn 11: What does it mean when we say a time series exhibits 'stationarity'?](#11)
- [Qn 12: What is the purpose of differencing in time series analysis?](#12)
- [Qn 13: What is seasonal differencing and when should it be applied?](#13)
- [Qn 14: What are residuals in the context of ARIMA modeling, and why are they important?](#14)
- [Qn 15: What does the Ljung-Box test evaluate in time series analysis?](#15)
- [Qn 16: What is the primary difference between ARMA and ARIMA models?](#16)
- [Qn 17: What is meant by the 'order of integration' in time series analysis?](#17)
- [Qn 18: What is the purpose of the Box-Jenkins methodology in time series analysis?](#18)
- [Qn 19: What information criterion is commonly used to select between different ARIMA models?](#19)
- [Qn 20: In the context of ARIMA residual analysis, what should a Q-Q plot ideally show?](#20)
- [Qn 21: What is the meaning of the 'exogenous variables' in the context of SARIMAX models?](#21)
- [Qn 22: Why might you perform a Box-Cox transformation before applying an ARIMA model?](#22)
- [Qn 23: What does it mean when an ARIMA model is said to be 'invertible'?](#23)
- [Qn 24: What is the difference between strong and weak stationarity in time series?](#24)
- [Qn 25: What is the primary purpose of the KPSS test in time series analysis?](#25)
- [Qn 26: What does Facebook Prophet use to model seasonality in time series data?](#26)
- [Qn 27: What are the three main components of a Facebook Prophet model?](#27)

---

### 1. Qn 01: What does the 'AR' component in ARIMA represent, and how does it capture patterns in time series data?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Autoregressive - using past values to predict future values

**Explanation:** The 'AR' in ARIMA stands for Autoregressive, which means the model uses past values of the time series to predict future values. Specifically, an AR(p) model uses p previous time steps as predictors. For example, in an AR(2) model, the current value is predicted using a linear combination of the previous two values, plus an error term. This component is particularly useful for capturing momentum or inertia in time series where recent values influence future values.


[Go to TOC](#table-of-contents)

</details>

---
### 2. Qn 02: What does the 'I' component in ARIMA represent, and why is it necessary?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Integrated - differencing to achieve stationarity

**Explanation:** The 'I' in ARIMA stands for Integrated, which refers to differencing the time series to achieve stationarity. Many time series have trends or seasonal patterns that make them non-stationary. The 'd' parameter in ARIMA(p,d,q) indicates how many times the data needs to be differenced to achieve stationarity. For example, if d=1, we take the difference between consecutive observations. This transformation is necessary because ARIMA models assume the underlying process is stationary, meaning its statistical properties do not change over time.


[Go to TOC](#table-of-contents)

</details>

---
### 3. Qn 03: What does the 'MA' component in ARIMA represent, and how does it differ from AR?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Moving Average - using past forecast errors in the model

**Explanation:** The 'MA' in ARIMA stands for Moving Average, which incorporates past forecast errors (residuals) into the model rather than past values of the time series itself. An MA(q) model uses the previous q forecast errors as predictors. This differs fundamentally from AR, which uses the actual past values. MA components capture the short-term reactions to past shocks or random events in the system. For example, an MA(1) model would use the forecast error from the previous time step to adjust the current prediction.


[Go to TOC](#table-of-contents)

</details>

---
### 4. Qn 04: How do you interpret the parameters p, d, and q in ARIMA(p,d,q)?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** p = AR order, d = differencing order, q = MA order

**Explanation:** In ARIMA(p,d,q), p represents the order of the autoregressive (AR) component, indicating how many lagged values of the series are included in the model. A higher p means more past values are used for prediction. The parameter d represents the degree of differencing required to make the series stationary, with d=1 meaning first difference, d=2 meaning second difference, etc. Finally, q is the order of the moving average (MA) component, indicating how many lagged forecast errors are included in the model. Together, these parameters define the structure of the ARIMA model and must be carefully selected based on the characteristics of the time series.


[Go to TOC](#table-of-contents)

</details>

---
### 5. Qn 05: What is the key assumption that must be satisfied before applying ARIMA models?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** The time series must be stationary

**Explanation:** The fundamental assumption for ARIMA models is that the time series is stationary or can be made stationary through differencing. A stationary time series has constant mean, variance, and autocorrelation structure over time. Without stationarity, the model cannot reliably learn patterns from the data. This is why the 'I' (Integrated) component exists in ARIMA - to transform non-stationary data through differencing. Analysts typically use statistical tests like the Augmented Dickey-Fuller (ADF) test to check for stationarity before applying ARIMA models.


[Go to TOC](#table-of-contents)

</details>

---
### 6. Qn 06: How can you determine the appropriate values for p and q in an ARIMA(p,d,q) model?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** By examining ACF and PACF plots

**Explanation:** The appropriate values for p and q in an ARIMA model can be determined by examining the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots of the stationary time series. For identifying the AR order (p), look for significant spikes in the PACF that cut off after lag p. For the MA order (q), look for significant spikes in the ACF that cut off after lag q. Additionally, information criteria like AIC (Akaike Information Criterion) or BIC (Bayesian Information Criterion) can be used to compare different model specifications and select the best combination of parameters.


[Go to TOC](#table-of-contents)

</details>

---
### 7. Qn 07: What is the purpose of the Augmented Dickey-Fuller (ADF) test in time series analysis?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** To test for stationarity

**Explanation:** The Augmented Dickey-Fuller (ADF) test is a statistical test used to determine whether a time series is stationary or not. The null hypothesis of the test is that the time series contains a unit root, implying it is non-stationary. If the p-value from the test is less than the significance level (typically 0.05), we reject the null hypothesis and conclude that the series is stationary. This test is crucial before applying ARIMA models because stationarity is a key assumption. The test includes lags of the differenced series to account for serial correlation, making it more robust than the simple Dickey-Fuller test.


[Go to TOC](#table-of-contents)

</details>

---
### 8. Qn 08: What is the difference between ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function)?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** ACF measures correlation between series and lagged values accounting for intermediate lags, PACF removes indirect correlation effects

**Explanation:** The ACF (Autocorrelation Function) measures the correlation between a time series and its lagged values, including both direct and indirect effects. It shows the correlation at each lag without controlling for correlations at shorter lags. In contrast, the PACF (Partial Autocorrelation Function) measures the correlation between a time series and its lagged values while controlling for the values of the time series at all shorter lags. This effectively removes the indirect correlation effects, showing only the direct relationship between observations separated by a specific lag. ACF helps identify MA(q) order, while PACF helps identify AR(p) order in ARIMA modeling.


[Go to TOC](#table-of-contents)

</details>

---
### 9. Qn 09: What additional component does SARIMAX add compared to ARIMA?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Seasonal components and exogenous variables

**Explanation:** SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) extends ARIMA by adding two important capabilities. First, it incorporates seasonal components, allowing the model to capture repeating patterns that occur at fixed intervals (like daily, weekly, or yearly seasonality). The seasonal component is specified with parameters (P,D,Q)m, where m is the seasonal period. Second, SARIMAX allows for exogenous variables (the 'X' part), which are external factors that can influence the time series but are not part of the series itself. These could include variables like temperature affecting energy consumption, or promotions affecting sales. This makes SARIMAX much more versatile than standard ARIMA for real-world applications with seasonal patterns and external influences.


[Go to TOC](#table-of-contents)

</details>

---
### 10. Qn 10: How are seasonality parameters represented in a SARIMA model?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** As (P,D,Q)m where m is the seasonal period

**Explanation:** In a SARIMA (Seasonal ARIMA) model, seasonality parameters are represented as (P,D,Q)m, where P is the seasonal autoregressive order, D is the seasonal differencing order, Q is the seasonal moving average order, and m is the number of periods in each season (the seasonal period). For example, in monthly data with yearly seasonality, m would be 12. In a SARIMA(1,1,1)(1,1,1)12 model, the non-seasonal components are (1,1,1) and the seasonal components are (1,1,1)12. The seasonal components operate at lag m, 2m, etc., capturing patterns that repeat every m periods. Seasonal differencing (D) involves subtracting the value from m periods ago, helping to remove seasonal non-stationarity.


[Go to TOC](#table-of-contents)

</details>

---
### 11. Qn 11: What does it mean when we say a time series exhibits 'stationarity'?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Its statistical properties remain constant over time

**Explanation:** A stationary time series has statistical properties that remain constant over time. Specifically, it has a constant mean, constant variance, and a constant autocorrelation structure. This means the process generating the time series is in statistical equilibrium. Stationarity is a crucial assumption for many time series models, including ARIMA, because it ensures that patterns learned from historical data will continue to be valid in the future. Non-stationary series might have trends (changing mean) or heteroscedasticity (changing variance), which can lead to unreliable forecasts if not properly addressed through transformations like differencing or variance stabilization.


[Go to TOC](#table-of-contents)

</details>

---
### 12. Qn 12: What is the purpose of differencing in time series analysis?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** To remove trends and achieve stationarity

**Explanation:** Differencing in time series analysis involves computing the differences between consecutive observations. The primary purpose is to remove trends and achieve stationarity, which is a key requirement for ARIMA modeling. First-order differencing (d=1) can eliminate linear trends by calculating Yt - Yt-1. If the series still shows non-stationarity after first differencing, second-order differencing (d=2) can be applied to remove quadratic trends. However, over-differencing can introduce unnecessary complexity and artificial patterns, so it's important to use statistical tests like the ADF test to determine the appropriate level of differencing needed.


[Go to TOC](#table-of-contents)

</details>

---
### 13. Qn 13: What is seasonal differencing and when should it be applied?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Calculating differences between observations separated by the seasonal period, applied when there's seasonal non-stationarity

**Explanation:** Seasonal differencing involves calculating differences between observations separated by the seasonal period (e.g., 12 months for monthly data with yearly seasonality). It's represented by the D parameter in SARIMA models and is applied when the time series exhibits seasonal non-stationarity, meaning the seasonal pattern changes over time. For example, with monthly data, seasonal differencing would compute Yt - Yt-12. This helps remove repeating seasonal patterns just as regular differencing removes trends. You should apply seasonal differencing when visual inspection shows persistent seasonal patterns after regular differencing, or when seasonal unit root tests indicate seasonal non-stationarity.


[Go to TOC](#table-of-contents)

</details>

---
### 14. Qn 14: What are residuals in the context of ARIMA modeling, and why are they important?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** The differences between observed and predicted values, important for diagnostic checking

**Explanation:** In ARIMA modeling, residuals are the differences between the observed values and the values predicted by the model. They represent the part of the data that the model couldn't explain. Residuals are crucial for diagnostic checking because a well-fitted ARIMA model should have residuals that resemble white noise - they should be uncorrelated, have zero mean, constant variance, and follow a normal distribution. If patterns remain in the residuals, it suggests the model hasn't captured all the systematic information in the time series. Common residual diagnostics include ACF/PACF plots of residuals, the Ljung-Box test for autocorrelation, and Q-Q plots for normality checking.


[Go to TOC](#table-of-contents)

</details>

---
### 15. Qn 15: What does the Ljung-Box test evaluate in time series analysis?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Whether residuals exhibit autocorrelation

**Explanation:** The Ljung-Box test is a statistical test used to evaluate whether residuals from a time series model exhibit autocorrelation. The null hypothesis is that the residuals are independently distributed (i.e., no autocorrelation). If the p-value is less than the significance level (typically 0.05), we reject the null hypothesis and conclude that the residuals contain significant autocorrelation, suggesting the model hasn't captured all the patterns in the data. The test examines multiple lags simultaneously, making it more comprehensive than just looking at individual autocorrelation values. A good ARIMA model should have residuals that pass the Ljung-Box test, indicating they approximate white noise.


[Go to TOC](#table-of-contents)

</details>

---
### 16. Qn 16: What is the primary difference between ARMA and ARIMA models?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** ARIMA includes differencing for non-stationary data, while ARMA requires stationary data

**Explanation:** The primary difference between ARMA (AutoRegressive Moving Average) and ARIMA (AutoRegressive Integrated Moving Average) models is that ARIMA includes a differencing step (the 'I' component) to handle non-stationary data. ARMA models combine autoregressive (AR) and moving average (MA) components but assume that the time series is already stationary. ARIMA extends this by first differencing the data d times to achieve stationarity before applying the ARMA model. This makes ARIMA more versatile for real-world time series that often contain trends. Essentially, an ARIMA(p,d,q) model is equivalent to applying an ARMA(p,q) model to a time series after differencing it d times.


[Go to TOC](#table-of-contents)

</details>

---
### 17. Qn 17: What is meant by the 'order of integration' in time series analysis?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** The number of times a series needs to be differenced to achieve stationarity

**Explanation:** The 'order of integration' refers to the number of times a time series needs to be differenced to achieve stationarity. It's represented by the parameter d in ARIMA(p,d,q) models. A series that requires differencing once (d=1) to become stationary is said to be integrated of order 1, or I(1). Similarly, a series requiring two differences is I(2). A naturally stationary series is I(0). The concept is important because it quantifies how persistent trends are in the data. Most economic and business time series are I(1), meaning they have stochastic trends that can be removed with first differencing. The order of integration can be determined using unit root tests like the Augmented Dickey-Fuller test.


[Go to TOC](#table-of-contents)

</details>

---
### 18. Qn 18: What is the purpose of the Box-Jenkins methodology in time series analysis?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** A systematic approach to identify, estimate, and validate ARIMA models

**Explanation:** The Box-Jenkins methodology is a systematic approach to identify, estimate, and validate ARIMA models for time series forecasting. It consists of three main stages: identification, estimation, and diagnostic checking. In the identification stage, you determine appropriate values for p, d, and q by analyzing ACF/PACF plots and using stationarity tests. In the estimation stage, you fit the selected ARIMA model to the data and estimate its parameters. In the diagnostic checking stage, you analyze residuals to ensure they resemble white noise and refine the model if needed. Box-Jenkins emphasizes iterative model building, where you cycle through these stages until you find an adequate model. This methodical approach helps ensure that the final model captures the data's patterns efficiently.


[Go to TOC](#table-of-contents)

</details>

---
### 19. Qn 19: What information criterion is commonly used to select between different ARIMA models?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** AIC (Akaike Information Criterion)

**Explanation:** The AIC (Akaike Information Criterion) is commonly used to select between different ARIMA models. It balances model fit against complexity by penalizing models with more parameters. The formula is AIC = -2log(L) + 2k, where L is the likelihood of the model and k is the number of parameters. A lower AIC value indicates a better model. When comparing ARIMA models with different p, d, and q values, analysts typically choose the model with the lowest AIC. Other similar criteria include BIC (Bayesian Information Criterion), which penalizes model complexity more heavily. These criteria help prevent overfitting by ensuring that additional parameters are only included if they substantially improve the model's fit to the data.


[Go to TOC](#table-of-contents)

</details>

---
### 20. Qn 20: In the context of ARIMA residual analysis, what should a Q-Q plot ideally show?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Points falling approximately along a straight line, indicating normally distributed residuals

**Explanation:** In ARIMA residual analysis, a Q-Q (Quantile-Quantile) plot should ideally show points falling approximately along a straight line. This indicates that the residuals follow a normal distribution, which is an assumption for valid statistical inference in ARIMA modeling. The Q-Q plot compares the quantiles of the residuals against the quantiles of a theoretical normal distribution. Deviations from the straight line suggest non-normality: a sigmoidal pattern indicates skewness, while an S-shaped curve suggests heavy or light tails compared to a normal distribution. Serious deviations might indicate model misspecification or the presence of outliers that could affect the reliability of confidence intervals and hypothesis tests for the model parameters.


[Go to TOC](#table-of-contents)

</details>

---
### 21. Qn 21: What is the meaning of the 'exogenous variables' in the context of SARIMAX models?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** External predictor variables that influence the time series but are not influenced by it

**Explanation:** In SARIMAX models, exogenous variables (the 'X' part) are external predictor variables that influence the time series being modeled but are not influenced by it. These are independent variables that provide additional information beyond what's contained in the past values of the time series itself. For example, when forecasting electricity demand, temperature might be an exogenous variable since it affects demand but isn't affected by it. Unlike the autoregressive components that use the series' own past values, exogenous variables inject outside information into the model. This can significantly improve forecast accuracy when the time series is known to be affected by measurable external factors. Mathematically, exogenous variables enter the SARIMAX equation as a regression component.


[Go to TOC](#table-of-contents)

</details>

---
### 22. Qn 22: Why might you perform a Box-Cox transformation before applying an ARIMA model?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** To stabilize variance and make the data more normally distributed

**Explanation:** A Box-Cox transformation is often performed before applying an ARIMA model to stabilize variance and make the data more normally distributed. Many time series exhibit heteroscedasticity (changing variance over time) or skewness, which can violate ARIMA assumptions. The Box-Cox transformation is a family of power transformations defined by the parameter λ: when λ=0, it's equivalent to a log transformation; when λ=1, it's essentially the original data (with a shift). The optimal λ value can be determined by maximizing the log-likelihood function. This transformation helps make the time series' variance more constant across time and its distribution more symmetric, leading to more reliable parameter estimates and prediction intervals in the ARIMA model.


[Go to TOC](#table-of-contents)

</details>

---
### 23. Qn 23: What does it mean when an ARIMA model is said to be 'invertible'?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** The MA component can be rewritten as an infinite AR process

**Explanation:** In time series analysis, when an ARIMA model is said to be 'invertible,' it means that its Moving Average (MA) component can be rewritten as an infinite Autoregressive (AR) process. This property ensures that the MA coefficients decrease in impact as we go further back in time, allowing the process to be approximated by a finite AR model. Invertibility is a mathematical property that ensures a unique MA representation and stable forecasting. Technically, for invertibility, the roots of the MA polynomial must lie outside the unit circle. Without invertibility, different MA models could produce identical autocorrelation patterns, making identification problematic. Invertibility is analogous to stationarity for AR processes and is checked during the model estimation phase.


[Go to TOC](#table-of-contents)

</details>

---
### 24. Qn 24: What is the difference between strong and weak stationarity in time series?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Weak stationarity requires constant mean and variance and time-invariant autocorrelation; strong stationarity requires the entire distribution to be time-invariant

**Explanation:** The distinction between strong (strict) and weak stationarity lies in how much of the data's statistical properties must remain constant over time. Weak stationarity, which is usually sufficient for ARIMA modeling, requires only that the mean and variance remain constant and that the autocorrelation function depends only on the lag between points, not their absolute position in time. In contrast, strong stationarity is more demanding, requiring that the entire joint probability distribution of the process remains unchanged when shifted in time. This means all higher moments (not just the first two) must be constant, and all multivariate distributions (not just bivariate correlations) must be time-invariant. In practice, analysts typically work with weak stationarity because it's easier to test for and sufficient for many applications.


[Go to TOC](#table-of-contents)

</details>

---
### 25. Qn 25: What is the primary purpose of the KPSS test in time series analysis?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** To test for stationarity with a null hypothesis of stationarity

**Explanation:** The KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test is used to test for stationarity in time series analysis, but unlike the ADF test, its null hypothesis is that the series is stationary. This reversal makes it a complementary test to ADF, which has a null hypothesis of non-stationarity. Using both tests together provides stronger evidence: if the ADF test rejects its null and the KPSS fails to reject its null, you have consistent evidence of stationarity. The KPSS test specifically tests whether the series can be described as stationary around a deterministic trend or has a unit root. A low p-value leads to rejecting the null, suggesting non-stationarity. This test is particularly useful for distinguishing between trend-stationary processes and difference-stationary processes.


[Go to TOC](#table-of-contents)

</details>

---
### 26. Qn 26: What does Facebook Prophet use to model seasonality in time series data?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Fourier series for multiple seasonal periods

**Explanation:** Facebook Prophet uses Fourier series to model seasonality in time series data. This approach represents seasonal patterns as a sum of sine and cosine terms of different frequencies, allowing for flexible modeling of complex seasonal patterns. Prophet can simultaneously model multiple seasonal periods (e.g., daily, weekly, and yearly seasonality) by using different Fourier series for each. The number of terms in each Fourier series (specified by the 'order' parameter) controls the flexibility of the seasonal component - higher orders capture more complex patterns but risk overfitting. This approach is particularly powerful because it can handle irregular time series and missing data better than traditional seasonal ARIMA models, which require regular time intervals.


[Go to TOC](#table-of-contents)

</details>

---
### 27. Qn 27: What are the three main components of a Facebook Prophet model?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Trend, seasonality, and holidays/events

**Explanation:** Facebook Prophet decomposes time series into three main components: trend, seasonality, and holidays/events. The trend component captures non-periodic changes, and can be modeled as either linear or logistic growth with automatic changepoint detection to accommodate trend changes. The seasonality component captures periodic patterns using Fourier series, and can simultaneously model multiple seasonal patterns (e.g., daily, weekly, annual). The holidays/events component accounts for irregular schedules and events that affect the time series but don't follow a seasonal pattern. Users can provide a custom list of holidays or events with their dates. By modeling these components separately and then adding them together, Prophet creates an interpretable forecast that can be easily understood and adjusted by analysts.


[Go to TOC](#table-of-contents)

</details>

---
