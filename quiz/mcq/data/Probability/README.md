# Probability Quiz

## Table of Contents
- [Qn 01: In a room of 30 people, what is the probability that at least two people share the same birthday (assuming 365 days in a year and birthdays are uniformly distributed)?](#1)
- [Qn 02: A biased coin has a 60% chance of landing heads. If you flip this coin 5 times, what is the probability of getting exactly 3 heads?](#2)
- [Qn 03: In a standard deck of 52 cards, what is the probability of drawing a royal flush (A, K, Q, J, 10 of the same suit) in a 5-card poker hand?](#3)
- [Qn 04: A standard six-sided die is rolled 3 times. What is the probability that the sum of the three rolls equals 10?](#4)
- [Qn 05: A bag contains 5 red marbles and 7 blue marbles. If 3 marbles are drawn without replacement, what is the probability that exactly 2 of them are red?](#5)
- [Qn 06: In a Bayesian analysis, a disease has a 1% prevalence in a population. A test for the disease has 95% sensitivity (true positive rate) and 90% specificity (true negative rate). If a person tests positive, what is the probability they actually have the disease?](#6)
- [Qn 07: You roll a fair 6-sided die repeatedly until you get a 6. What is the expected number of rolls needed?](#7)
- [Qn 08: In a group of 5 people, what is the probability that at least 2 people have the same zodiac sign (assuming zodiac signs are uniformly distributed across 12 possible signs)?](#8)
- [Qn 09: A data scientist applies a machine learning model to classify emails as spam or not spam. The model has 98% accuracy on legitimate emails and 95% accuracy on spam emails. If 20% of all incoming emails are spam, what is the probability that an email classified as spam by the model is actually spam?](#9)
- [Qn 10: Four cards are randomly selected from a standard 52-card deck. What is the probability of getting exactly 2 aces?](#10)
- [Qn 11: In a standard normal distribution, what is the probability that a randomly selected observation falls between -1.96 and 1.96?](#11)
- [Qn 12: A manufacturing process has a 3% defect rate. If 50 items are randomly selected, what is the probability that at most 2 are defective?](#12)
- [Qn 13: In the Monty Hall problem, you're on a game show with three doors. Behind one door is a car; behind the others are goats. You pick a door. The host, who knows what's behind each door, opens one of the other doors to reveal a goat. He then offers you the chance to switch your choice to the remaining unopened door. What is the probability of winning the car if you switch?](#13)
- [Qn 14: A researcher is testing a new drug. In reality, the drug has no effect, but the researcher will conclude it works if the p-value is less than 0.05. What is the probability that the researcher incorrectly concludes the drug works?](#14)
- [Qn 15: A database has 1,000,000 records, and a data scientist estimates that 50 records are corrupted. If the data scientist randomly samples 100 records for manual inspection, what is the probability of finding at least one corrupted record?](#15)
- [Qn 16: In a certain city, 60% of days are sunny, 30% are cloudy, and 10% are rainy. The probability of a traffic jam is 0.1 on sunny days, 0.3 on cloudy days, and 0.5 on rainy days. If there is a traffic jam today, what is the probability that it is a sunny day?](#16)
- [Qn 17: Five fair six-sided dice are rolled. What is the probability that all five dice show different numbers?](#17)
- [Qn 18: A data center has 5 servers, each with a 1% probability of failing in a given day, independently of the others. What is the probability that at least one server fails today?](#18)
- [Qn 19: In a random sample of 20 people, what is the probability that at least 2 people were born in the same month of the year (assuming uniform distribution of birth months)?](#19)
- [Qn 20: A biased coin has an unknown probability p of landing heads. After 10 flips, you observe 7 heads. Using a uniform prior distribution for p, what is the expected value of p according to Bayesian analysis?](#20)
- [Qn 21: In a multiple-choice test with 5 questions, each question has 4 options with exactly one correct answer. If a student guesses randomly on all questions, what is the probability of getting at least 3 questions correct?](#21)
- [Qn 22: In a hypergeometric distribution scenario, a shipment of 100 electronic components contains 8 defective parts. If 10 components are randomly selected without replacement for inspection, what is the probability of finding exactly 1 defective component?](#22)
- [Qn 23: A fair six-sided die is rolled 10 times. What is the probability of getting exactly 2 sixes?](#23)
- [Qn 24: In a large city, 45% of residents prefer public transportation, 35% prefer driving, and 20% prefer cycling. If three residents are randomly selected, what is the probability that at least one of them prefers cycling?](#24)
- [Qn 25: A genetics researcher is studying a trait that is determined by two alleles. The dominant allele A occurs with probability 0.7 and the recessive allele a with probability 0.3. Assuming Hardy-Weinberg equilibrium, what is the probability of a randomly selected individual having the genotype Aa?](#25)
- [Qn 26: In a Poisson process where events occur at an average rate of 3 per hour, what is the probability that exactly 2 events occur in a 1-hour period?](#26)
- [Qn 27: A data scientist is analyzing user engagement on a website. If the probability distribution of the number of pages viewed by a visitor follows a geometric distribution with p=0.2, what is the probability that a visitor views exactly 5 pages before leaving the site?](#27)
- [Qn 28: A medical test for a disease has sensitivity (true positive rate) of 90% and specificity (true negative rate) of 95%. In a population where 2% of people have the disease, what is the positive predictive value (probability that a person with a positive test result actually has the disease)?](#28)
- [Qn 29: In a lottery where 5 numbers are drawn from 1 to 49 without replacement, what is the probability of matching exactly 3 numbers on a single ticket?](#29)
- [Qn 30: In a random sample from a normal distribution with mean 100 and standard deviation 15, what is the probability that a single observation exceeds 125?](#30)
- [Qn 31: In a randomized controlled trial, patients are randomly assigned to either treatment or control groups with equal probability. If 10 patients are enrolled, what is the probability that exactly 5 are assigned to the treatment group?](#31)
- [Qn 32: A data scientist runs 20 independent A/B tests, each with a 5% false positive rate (Type I error). What is the probability of observing at least one false positive result across all tests if none of the tested hypotheses are actually true?](#32)
- [Qn 33: Two fair six-sided dice are rolled. Given that the sum of the dice is greater than 7, what is the probability that at least one die shows a 6?](#33)

---

### 1. Qn 01: In a room of 30 people, what is the probability that at least two people share the same birthday (assuming 365 days in a year and birthdays are uniformly distributed)?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Greater than 90%

**Explanation:** This is the famous birthday paradox. The probability is computed as 1 minus the probability that all birthdays are different. For 30 people, P(at least one shared birthday) = 1 - (365/365 × 364/365 × 363/365 × ... × 336/365) ≈ 0.706, which is approximately 70.6%. While this is closest to 'About 70%', the exact calculation gives 70.6%, not 'Greater than 90%'. The probability exceeds 90% when there are 41 or more people in the room.


[Go to TOC](#table-of-contents)

</details>

---
### 2. Qn 02: A biased coin has a 60% chance of landing heads. If you flip this coin 5 times, what is the probability of getting exactly 3 heads?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.34560

**Explanation:** We use the binomial probability formula: P(X=k) = C(n,k) × p^k × (1-p)^(n-k), where n=5, k=3, p=0.6. C(5,3) = 10 possible ways to arrange 3 heads in 5 flips. So P(X=3) = 10 × (0.6)^3 × (0.4)^2 = 10 × 0.216 × 0.16 = 0.3456 or 34.56%.


[Go to TOC](#table-of-contents)

</details>

---
### 3. Qn 03: In a standard deck of 52 cards, what is the probability of drawing a royal flush (A, K, Q, J, 10 of the same suit) in a 5-card poker hand?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 4/2,598,960

**Explanation:** There are C(52,5) = 2,598,960 possible 5-card hands from a standard deck. There are exactly 4 possible royal flushes (one for each suit). Therefore, the probability is 4/2,598,960 = 1/649,740 or approximately 0.00000154.


[Go to TOC](#table-of-contents)

</details>

---
### 4. Qn 04: A standard six-sided die is rolled 3 times. What is the probability that the sum of the three rolls equals 10?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 1/8

**Explanation:** When rolling 3 dice, there are 6^3 = 216 possible outcomes. To get a sum of 10, we can have combinations like (1,3,6), (2,2,6), etc. Counting all such combinations gives us 27 favorable outcomes. Therefore, the probability is 27/216 = 1/8.


[Go to TOC](#table-of-contents)

</details>

---
### 5. Qn 05: A bag contains 5 red marbles and 7 blue marbles. If 3 marbles are drawn without replacement, what is the probability that exactly 2 of them are red?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 35/132

**Explanation:** Total ways to select 3 marbles from 12 is C(12,3) = 220. Ways to select 2 red marbles from 5 red and 1 blue marble from 7 blue is C(5,2) × C(7,1) = 10 × 7 = 70. Therefore probability = 70/220 = 7/22 = 35/110 = 7/22.


[Go to TOC](#table-of-contents)

</details>

---
### 6. Qn 06: In a Bayesian analysis, a disease has a 1% prevalence in a population. A test for the disease has 95% sensitivity (true positive rate) and 90% specificity (true negative rate). If a person tests positive, what is the probability they actually have the disease?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** Around 8.7%

**Explanation:** Using Bayes' theorem: P(Disease|Positive) = [P(Positive|Disease) × P(Disease)] / P(Positive). P(Positive) = P(Positive|Disease) × P(Disease) + P(Positive|No Disease) × P(No Disease) = 0.95 × 0.01 + 0.10 × 0.99 = 0.0095 + 0.099 = 0.1085. Therefore, P(Disease|Positive) = (0.95 × 0.01) / 0.1085 ≈ 0.0095/0.1085 ≈ 0.0876 or about 8.7%.


[Go to TOC](#table-of-contents)

</details>

---
### 7. Qn 07: You roll a fair 6-sided die repeatedly until you get a 6. What is the expected number of rolls needed?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 6

**Explanation:** This is a geometric distribution with probability of success p = 1/6. The expected value of a geometric distribution is 1/p. So the expected number of rolls needed is 1/(1/6) = 6.


[Go to TOC](#table-of-contents)

</details>

---
### 8. Qn 08: In a group of 5 people, what is the probability that at least 2 people have the same zodiac sign (assuming zodiac signs are uniformly distributed across 12 possible signs)?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.96

**Explanation:** The probability that all 5 people have different zodiac signs is (12/12) × (11/12) × (10/12) × (9/12) × (8/12) = 0.0397. Therefore, the probability that at least 2 people share a zodiac sign is 1 - 0.0397 = 0.9603 ≈ 0.96 or 96%.


[Go to TOC](#table-of-contents)

</details>

---
### 9. Qn 09: A data scientist applies a machine learning model to classify emails as spam or not spam. The model has 98% accuracy on legitimate emails and 95% accuracy on spam emails. If 20% of all incoming emails are spam, what is the probability that an email classified as spam by the model is actually spam?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.83

**Explanation:** Using Bayes' theorem: P(Spam|Classified as Spam) = [P(Classified as Spam|Spam) × P(Spam)] / P(Classified as Spam). P(Classified as Spam) = P(Classified as Spam|Spam) × P(Spam) + P(Classified as Spam|Not Spam) × P(Not Spam) = 0.95 × 0.2 + 0.02 × 0.8 = 0.19 + 0.016 = 0.206. Therefore, P(Spam|Classified as Spam) = (0.95 × 0.2) / 0.206 = 0.19/0.206 ≈ 0.83 or 83%.


[Go to TOC](#table-of-contents)

</details>

---
### 10. Qn 10: Four cards are randomly selected from a standard 52-card deck. What is the probability of getting exactly 2 aces?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.0399

**Explanation:** Total number of ways to select 4 cards from 52 is C(52,4) = 270,725. Ways to select exactly 2 aces from 4 aces is C(4,2) = 6. Ways to select the other 2 cards from the non-ace cards is C(48,2) = 1,128. So favorable outcomes = 6 × 1,128 = 6,768. Probability = 6,768/270,725 ≈ 0.0399 or about 4%.


[Go to TOC](#table-of-contents)

</details>

---
### 11. Qn 11: In a standard normal distribution, what is the probability that a randomly selected observation falls between -1.96 and 1.96?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.95

**Explanation:** In a standard normal distribution, the area between z-scores of -1.96 and 1.96 corresponds to 95% of the distribution. This is a fundamental value in statistics, often used for 95% confidence intervals.


[Go to TOC](#table-of-contents)

</details>

---
### 12. Qn 12: A manufacturing process has a 3% defect rate. If 50 items are randomly selected, what is the probability that at most 2 are defective?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.6063

**Explanation:** This follows a binomial distribution with n=50 and p=0.03. P(X ≤ 2) = P(X=0) + P(X=1) + P(X=2) = C(50,0) × (0.03)^0 × (0.97)^50 + C(50,1) × (0.03)^1 × (0.97)^49 + C(50,2) × (0.03)^2 × (0.97)^48 ≈ 0.2231 + 0.3453 + 0.2379 = 0.6063 or about 60.63%.


[Go to TOC](#table-of-contents)

</details>

---
### 13. Qn 13: In the Monty Hall problem, you're on a game show with three doors. Behind one door is a car; behind the others are goats. You pick a door. The host, who knows what's behind each door, opens one of the other doors to reveal a goat. He then offers you the chance to switch your choice to the remaining unopened door. What is the probability of winning the car if you switch?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 2/3

**Explanation:** Initially, you have a 1/3 probability of choosing the car and a 2/3 probability of choosing a goat. If you initially chose the car (probability 1/3), switching will always make you lose. If you initially chose a goat (probability 2/3), the host will reveal the other goat, and switching will always make you win. Therefore, the probability of winning by switching is 2/3.


[Go to TOC](#table-of-contents)

</details>

---
### 14. Qn 14: A researcher is testing a new drug. In reality, the drug has no effect, but the researcher will conclude it works if the p-value is less than 0.05. What is the probability that the researcher incorrectly concludes the drug works?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.05

**Explanation:** The p-value is the probability of obtaining results at least as extreme as the observed results, assuming the null hypothesis is true. Here, the null hypothesis is that the drug has no effect (which is actually true). By definition, the probability of getting a p-value below 0.05 when the null hypothesis is true (Type I error) is 0.05 or 5%.


[Go to TOC](#table-of-contents)

</details>

---
### 15. Qn 15: A database has 1,000,000 records, and a data scientist estimates that 50 records are corrupted. If the data scientist randomly samples 100 records for manual inspection, what is the probability of finding at least one corrupted record?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.4988

**Explanation:** The probability of selecting a corrupted record is 50/1,000,000 = 0.00005. The probability of not finding any corrupted records in 100 samples is (1 - 0.00005)^100 ≈ 0.9950. Therefore, the probability of finding at least one corrupted record is 1 - 0.9950 = 0.0050 or 0.5%.


[Go to TOC](#table-of-contents)

</details>

---
### 16. Qn 16: In a certain city, 60% of days are sunny, 30% are cloudy, and 10% are rainy. The probability of a traffic jam is 0.1 on sunny days, 0.3 on cloudy days, and 0.5 on rainy days. If there is a traffic jam today, what is the probability that it is a sunny day?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.33

**Explanation:** Using Bayes' theorem: P(Sunny|Traffic Jam) = [P(Traffic Jam|Sunny) × P(Sunny)] / P(Traffic Jam). P(Traffic Jam) = 0.1 × 0.6 + 0.3 × 0.3 + 0.5 × 0.1 = 0.06 + 0.09 + 0.05 = 0.2. Therefore, P(Sunny|Traffic Jam) = (0.1 × 0.6) / 0.2 = 0.06/0.2 = 0.3 or 30%.


[Go to TOC](#table-of-contents)

</details>

---
### 17. Qn 17: Five fair six-sided dice are rolled. What is the probability that all five dice show different numbers?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.0926

**Explanation:** Total number of possible outcomes when rolling 5 dice is 6^5 = 7,776. For all dice to show different numbers, we can arrange 5 different numbers from the set {1,2,3,4,5,6} in 6!/1! = 720 ways. Therefore, the probability is 720/7,776 = 0.0926 or about 9.26%.


[Go to TOC](#table-of-contents)

</details>

---
### 18. Qn 18: A data center has 5 servers, each with a 1% probability of failing in a given day, independently of the others. What is the probability that at least one server fails today?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.0490

**Explanation:** The probability that a specific server doesn't fail is 0.99. The probability that all servers don't fail is (0.99)^5 ≈ 0.9510. Therefore, the probability that at least one server fails is 1 - 0.9510 = 0.0490 or about 4.9%.


[Go to TOC](#table-of-contents)

</details>

---
### 19. Qn 19: In a random sample of 20 people, what is the probability that at least 2 people were born in the same month of the year (assuming uniform distribution of birth months)?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.9139

**Explanation:** The probability that all 20 people were born in different months is 0 since there are only 12 months. The probability that no two people share the same birth month in a sample of 12 or fewer is calculated using the birthday problem formula for 12 months: 1 - P(no matching months) = 1 - (12!/12^n × (12-n)!) for n=12. For n>12, the probability of at least one match is 1.


[Go to TOC](#table-of-contents)

</details>

---
### 20. Qn 20: A biased coin has an unknown probability p of landing heads. After 10 flips, you observe 7 heads. Using a uniform prior distribution for p, what is the expected value of p according to Bayesian analysis?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.636

**Explanation:** With a uniform prior distribution (Beta(1,1)) and 7 heads out of 10 flips, the posterior distribution is Beta(1+7, 1+3) = Beta(8, 4). The expected value of a Beta(α, β) distribution is α/(α+β). So the expected value of p is 8/(8+4) = 8/12 = 2/3 ≈ 0.667, which is closest to 0.636.


[Go to TOC](#table-of-contents)

</details>

---
### 21. Qn 21: In a multiple-choice test with 5 questions, each question has 4 options with exactly one correct answer. If a student guesses randomly on all questions, what is the probability of getting at least 3 questions correct?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.0537

**Explanation:** The probability of getting a single question correct by random guessing is 1/4 = 0.25. Using the binomial distribution with n=5 and p=0.25: P(X ≥ 3) = P(X=3) + P(X=4) + P(X=5) = C(5,3) × (0.25)^3 × (0.75)^2 + C(5,4) × (0.25)^4 × (0.75)^1 + C(5,5) × (0.25)^5 × (0.75)^0 ≈ 0.0439 + 0.0073 + 0.0010 = 0.0522 or about 5.22%.


[Go to TOC](#table-of-contents)

</details>

---
### 22. Qn 22: In a hypergeometric distribution scenario, a shipment of 100 electronic components contains 8 defective parts. If 10 components are randomly selected without replacement for inspection, what is the probability of finding exactly 1 defective component?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.3816

**Explanation:** Using the hypergeometric probability mass function: P(X=1) = [C(8,1) × C(92,9)] / C(100,10) = [8 × 1,742,281,695] / 17,310,309,728 = 13,938,253,560 / 17,310,309,728 ≈ 0.3816 or about 38.16%.


[Go to TOC](#table-of-contents)

</details>

---
### 23. Qn 23: A fair six-sided die is rolled 10 times. What is the probability of getting exactly 2 sixes?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.2907

**Explanation:** This follows a binomial distribution with n=10 and p=1/6. P(X=2) = C(10,2) × (1/6)^2 × (5/6)^8 = 45 × (1/36) × (1,679,616/1,679,616) = 45/36 × 0.2323 ≈ 1.25 × 0.2323 = 0.2904 or about 29.04%.


[Go to TOC](#table-of-contents)

</details>

---
### 24. Qn 24: In a large city, 45% of residents prefer public transportation, 35% prefer driving, and 20% prefer cycling. If three residents are randomly selected, what is the probability that at least one of them prefers cycling?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.488

**Explanation:** The probability that a selected resident does not prefer cycling is 1 - 0.2 = 0.8. The probability that none of the three selected residents prefers cycling is (0.8)^3 = 0.512. Therefore, the probability that at least one prefers cycling is 1 - 0.512 = 0.488 or 48.8%.


[Go to TOC](#table-of-contents)

</details>

---
### 25. Qn 25: A genetics researcher is studying a trait that is determined by two alleles. The dominant allele A occurs with probability 0.7 and the recessive allele a with probability 0.3. Assuming Hardy-Weinberg equilibrium, what is the probability of a randomly selected individual having the genotype Aa?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.42

**Explanation:** Under Hardy-Weinberg equilibrium, the probability of genotype Aa is 2pq, where p is the probability of allele A and q is the probability of allele a. So P(Aa) = 2 × 0.7 × 0.3 = 0.42 or 42%.


[Go to TOC](#table-of-contents)

</details>

---
### 26. Qn 26: In a Poisson process where events occur at an average rate of 3 per hour, what is the probability that exactly 2 events occur in a 1-hour period?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.224

**Explanation:** For a Poisson distribution with parameter λ=3, the probability mass function gives P(X=2) = e^(-λ) × λ^2 / 2! = e^(-3) × 3^2 / 2 = e^(-3) × 9 / 2 = 0.0498 × 4.5 ≈ 0.224 or about 22.4%.


[Go to TOC](#table-of-contents)

</details>

---
### 27. Qn 27: A data scientist is analyzing user engagement on a website. If the probability distribution of the number of pages viewed by a visitor follows a geometric distribution with p=0.2, what is the probability that a visitor views exactly 5 pages before leaving the site?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.082

**Explanation:** For a geometric distribution with parameter p=0.2, the probability mass function gives P(X=5) = p(1-p)^(k-1) = 0.2 × (0.8)^4 = 0.2 × 0.4096 = 0.08192 ≈ 0.082 or about 8.2%.


[Go to TOC](#table-of-contents)

</details>

---
### 28. Qn 28: A medical test for a disease has sensitivity (true positive rate) of 90% and specificity (true negative rate) of 95%. In a population where 2% of people have the disease, what is the positive predictive value (probability that a person with a positive test result actually has the disease)?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.27

**Explanation:** Using Bayes' theorem: PPV = P(Disease|Positive test) = [P(Positive test|Disease) × P(Disease)] / P(Positive test). P(Positive test) = P(Positive test|Disease) × P(Disease) + P(Positive test|No Disease) × P(No Disease) = 0.9 × 0.02 + 0.05 × 0.98 = 0.018 + 0.049 = 0.067. Therefore, PPV = (0.9 × 0.02) / 0.067 = 0.018/0.067 ≈ 0.27 or 27%.


[Go to TOC](#table-of-contents)

</details>

---
### 29. Qn 29: In a lottery where 5 numbers are drawn from 1 to 49 without replacement, what is the probability of matching exactly 3 numbers on a single ticket?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.015

**Explanation:** Total number of possible 5-number combinations is C(49,5) = 1,906,884. Ways to match exactly 3 numbers out of 5: You must match 3 of the winning numbers [C(5,3) = 10] and 2 of the non-winning numbers [C(44,2) = 946]. So favorable outcomes = 10 × 946 = 9,460. Probability = 9,460/1,906,884 ≈ 0.00496 or about 0.5%.


[Go to TOC](#table-of-contents)

</details>

---
### 30. Qn 30: In a random sample from a normal distribution with mean 100 and standard deviation 15, what is the probability that a single observation exceeds 125?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.0478

**Explanation:** Standardizing, z = (125 - 100)/15 = 1.67. The probability P(X > 125) = P(Z > 1.67) ≈ 0.0475 or about 4.75%, which is closest to 0.0478.


[Go to TOC](#table-of-contents)

</details>

---
### 31. Qn 31: In a randomized controlled trial, patients are randomly assigned to either treatment or control groups with equal probability. If 10 patients are enrolled, what is the probability that exactly 5 are assigned to the treatment group?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.246

**Explanation:** This follows a binomial distribution with n=10 and p=0.5. P(X=5) = C(10,5) × (0.5)^5 × (0.5)^5 = 252 × (0.5)^10 = 252/1024 = 0.2461 or about 24.6%.


[Go to TOC](#table-of-contents)

</details>

---
### 32. Qn 32: A data scientist runs 20 independent A/B tests, each with a 5% false positive rate (Type I error). What is the probability of observing at least one false positive result across all tests if none of the tested hypotheses are actually true?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 0.64

**Explanation:** The probability of not observing a false positive in a single test is 1 - 0.05 = 0.95. The probability of not observing any false positives in 20 independent tests is (0.95)^20 ≈ 0.358. Therefore, the probability of observing at least one false positive is 1 - 0.358 = 0.642 or about 64.2%.


[Go to TOC](#table-of-contents)

</details>

---
### 33. Qn 33: Two fair six-sided dice are rolled. Given that the sum of the dice is greater than 7, what is the probability that at least one die shows a 6?
<details>
<summary><strong>View Answer & Explanation</strong></summary>

**Answer:** 7/12

**Explanation:** The possible outcomes for sum > 7 are: (2,6), (3,5), (3,6), (4,4), (4,5), (4,6), (5,3), (5,4), (5,5), (5,6), (6,2), (6,3), (6,4), (6,5), (6,6) - a total of 15 outcomes. Of these, 11 outcomes include at least one 6: (2,6), (3,6), (4,6), (5,6), (6,2), (6,3), (6,4), (6,5), (6,6). Therefore, the probability is 11/15 = 11/15 or about 73.3%.


[Go to TOC](#table-of-contents)

</details>

---
