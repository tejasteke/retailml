#  Retail Sales Analytics & Machine Learning Predictor

An interactive, premium web dashboard that combines exploratory data analysis (EDA) and machine learning models trained on the retail sales dataset. 

This project is structured specifically to be **GitHub Pages Ready** — the entire analytical interface and predictive simulators run directly in the browser with no server-side configuration, making it fully deployable on public GitHub hosting.

---

##  Live Demo & Deployment Guide (GitHub Pages)

You can host this interactive dashboard for free on GitHub Pages in **3 simple steps**:

1. **Create a GitHub Repository**: Create a new public repository on GitHub (e.g. `retail-sales-ml`).
2. **Push the Code**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of retail sales dashboard"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   git push -u origin main
   ```
3. **Enable GitHub Pages**:
   - Go to your repository settings on GitHub.
   - Select **Pages** from the sidebar.
   - Under *Build and deployment*, set the source to **Deploy from a branch**.
   - Select the `main` branch and root `/` folder, then click **Save**.
   
Your interactive dashboard will be live at `https://YOUR_USERNAME.github.io/YOUR_REPOSITORY/` within a few minutes!

---

##  Running the Dashboard Locally

To make running this dashboard as simple as possible, we compile all analytical and machine learning data into `assets/data/data.js`. Because the data is loaded directly as a JavaScript module script:

*   **Double-click `index.html`**: You can open `index.html` directly in your web browser (via double-clicking the file in File Explorer) without encountering any CORS security errors.
*   **Or run a local server**: If you prefer, you can host it locally using Python:
    ```bash
    python -m http.server 8000
    ```
    Then visit **[http://localhost:8000](http://localhost:8000)** in your browser.

---

##  Repository Structure
```
├── index.html                  # Core HTML dashboard structure
├── style.css                   # Modern dark glassmorphic stylesheet
├── app.js                      # Dynamic UI charts (Chart.js) & regression estimator
├── data_analysis.py            # Python script: Data cleaning, EDA & Matplotlib exports
├── machine_learning.py         # Python script: K-Means clustering, Ridge training & export
├── assets/
│   ├── data/
│   │   ├── eda_results.json    # JSON database for frontend queries
│   │   └── ml_results.json     # Trained regression coefficients & cluster centroids
│   └── plots/                  # Static Seaborn plots exported from Python
└── README.md                   # Project documentation
```

---

##  Insights from the EDA Journey

Here are the key takeaways from our Python-based data analysis answering your 7 questions:

1. **Age and Gender Influence**: Average spending stays consistently in the range of **$200 - $250** per transaction across all age groups (18 to 65+) and is split almost 50/50 between male and female shoppers. Demographics have minimal direct impact on the scale of a transaction.
2. **Temporal Patterns**: Sales do not show strong day-of-week biases, indicating steady weekly transaction rates. Monthly sales show mild peaks and troughs corresponding to seasonal demand variations.
3. **Category Appeal**: Electronics, Clothing, and Beauty are extremely balanced, holding a roughly **33.3%** revenue share each. Electronics leads slightly in terms of average unit price, but Clothing and Beauty match it in sheer quantity of transactions.
4. **Age, Spending & Product Relationship**: The correlation coefficient between Age and Total Amount is **-0.0606** (virtually zero). Furthermore, all age brackets purchase Beauty, Clothing, and Electronics in near-equal ratios. Age is not a strong determining factor for product category preferences.
5. **Seasonal Habits**: Shopping behavior shows mild adjustments during seasons. However, the generalized category labels maintain steady volumes across Spring, Summer, Autumn, and Winter.
6. **Quantity bought per Transaction**: Average transaction spending rises linearly with item count (1 to 4 units). However, unit prices remain flat across different volumes, indicating no volume discounts or bulk product shifts.
7. **Price Distribution**: Products span uniformly across all price levels (from $25 up to $500). Each product category contains a similar spread of cheap, moderate, and high-end items.

---

##  Machine Learning Model Explanations

### 1. Customer Segmentation (Unsupervised K-Means)
The model groups shoppers into 4 distinct segments based on transaction history (`Age`, `Total Amount`, `Quantity`):
* **Mature Budget Shoppers**: Older customers buying single items with lower unit prices. Highly budget-conscious.
* **Younger Moderate Shoppers**: Younger customers buying basic items, showing moderate spending habits.
* **Premium Value Buyers**: Shoppers buying high-priced items, resulting in high transaction totals despite buying few items.
* **High-Volume Spenders**: Top tier customers who buy multiple items in a single transaction, yielding high transaction totals.

*Our dashboard uses a client-side vector-distance calculator that scales your input features and assigns you to a segment in real-time.*

### 2. Purchase Estimator (Ridge Regression)
We trained Ridge Regression models in Python to predict the **Quantity** and **Total Spend** of any transaction.
* **Predictors**: Age, Gender (one-hot), Product Category (one-hot), Month, Year, Day of Week.
* **Model Equation**: $\text{Prediction} = \text{Intercept} + \sum (\text{Feature}_i \times \text{Weight}_i)$
* **Data Science Insight**: R² scores near 0 on this dataset represent an important real-world finding: **demographics and date features do not strongly predict transaction size**. The dataset's uniform simulation means retail managers must capture additional features (like browsing history or loyalty tier) to make high-accuracy predictions. The estimator displays the mathematical baseline weights.
