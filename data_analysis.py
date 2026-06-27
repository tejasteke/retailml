import os
import json
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics for plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def main():
    print("Step 1: Downloading/Retrieving retail sales dataset...")
    # This downloads or loads the dataset from the cache
    dataset_dir = kagglehub.dataset_download("mohammadtalib786/retail-sales-dataset")
    csv_path = os.path.join(dataset_dir, "retail_sales_dataset.csv")
    print(f"Dataset path: {csv_path}")

    # Load data
    df = pd.read_csv(csv_path)

    print("Step 2: Cleaning and preprocessing data...")
    # Standardize column headers and strings
    df.columns = df.columns.str.strip()
    df['Gender'] = df['Gender'].str.strip()
    df['Product Category'] = df['Product Category'].str.strip()
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.strftime('%B')
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['DayOfWeekNum'] = df['Date'].dt.dayofweek # 0=Monday, 6=Sunday
    df['Season'] = df['Month'].apply(get_season)

    # Bin Age into standard groups
    age_bins = [0, 25, 35, 45, 55, 65, 100]
    age_labels = ['Under 25', '25-34', '35-44', '45-54', '55-64', '65+']
    df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

    # Setup directories
    os.makedirs('assets/plots', exist_ok=True)
    os.makedirs('assets/data', exist_ok=True)

    # EDA results dictionary to export to JSON
    eda_data = {}

    print("Step 3: Conducting analysis & generating plots...")

    # --- Q1: How does customer age and gender influence their purchasing behavior? ---
    print("- Question 1...")
    q1_grouped = df.groupby(['Gender', 'Age Group'], observed=False).agg(
        Total_Spend=('Total Amount', 'sum'),
        Avg_Spend=('Total Amount', 'mean'),
        Total_Qty=('Quantity', 'sum'),
        Avg_Qty=('Quantity', 'mean'),
        Transaction_Count=('Transaction ID', 'count')
    ).reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=q1_grouped, x='Age Group', y='Avg_Spend', hue='Gender', palette='muted')
    plt.title('Average Spending by Age Group and Gender')
    plt.ylabel('Average Spend ($)')
    plt.xlabel('Age Group')
    plt.tight_layout()
    plt.savefig('assets/plots/q1_age_gender_spending.png', dpi=150)
    plt.close()

    # Save summary data
    eda_data['q1_age_gender'] = q1_grouped.to_dict(orient='records')

    # --- Q2: Are there discernible patterns in sales across different time periods? ---
    print("- Question 2...")
    # Monthly sales trend
    q2_monthly = df.groupby(['Year', 'Month', 'MonthName']).agg(
        Total_Sales=('Total Amount', 'sum'),
        Transaction_Count=('Transaction ID', 'count')
    ).reset_index().sort_values(['Year', 'Month'])
    q2_monthly['YearMonth'] = q2_monthly['Year'].astype(str) + '-' + q2_monthly['Month'].astype(str).str.zfill(2)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=q2_monthly, x='YearMonth', y='Total_Sales', marker='o', color='purple', linewidth=2.5)
    plt.title('Monthly Sales Trend (Total Amount)')
    plt.ylabel('Total Sales ($)')
    plt.xlabel('Time Period (YYYY-MM)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('assets/plots/q2_monthly_sales.png', dpi=150)
    plt.close()

    # Day of Week trend
    q2_dayofweek = df.groupby(['DayOfWeekNum', 'DayOfWeek']).agg(
        Total_Sales=('Total Amount', 'sum'),
        Avg_Sales=('Total Amount', 'mean'),
        Transaction_Count=('Transaction ID', 'count')
    ).reset_index().sort_values('DayOfWeekNum')

    plt.figure(figsize=(9, 5))
    sns.barplot(data=q2_dayofweek, x='DayOfWeek', y='Total_Sales', palette='coolwarm')
    plt.title('Total Sales by Day of the Week')
    plt.ylabel('Total Sales ($)')
    plt.xlabel('Day of Week')
    plt.tight_layout()
    plt.savefig('assets/plots/q2_dayofweek_sales.png', dpi=150)
    plt.close()

    eda_data['q2_monthly'] = q2_monthly.to_dict(orient='records')
    eda_data['q2_dayofweek'] = q2_dayofweek.to_dict(orient='records')

    # --- Q3: Which product categories hold the highest appeal among customers? ---
    print("- Question 3...")
    q3_categories = df.groupby('Product Category').agg(
        Total_Sales=('Total Amount', 'sum'),
        Total_Qty=('Quantity', 'sum'),
        Transaction_Count=('Transaction ID', 'count'),
        Avg_Price_Per_Unit=('Price per Unit', 'mean')
    ).reset_index()

    plt.figure(figsize=(8, 6))
    plt.pie(q3_categories['Total_Sales'], labels=q3_categories['Product Category'], autopct='%1.1f%%', 
            colors=sns.color_palette('pastel')[0:3], startangle=140, textprops={'fontsize': 14})
    plt.title('Share of Total Sales by Product Category')
    plt.tight_layout()
    plt.savefig('assets/plots/q3_product_share.png', dpi=150)
    plt.close()

    eda_data['q3_categories'] = q3_categories.to_dict(orient='records')

    # --- Q4: What are the relationships between age, spending, and product preferences? ---
    print("- Question 4...")
    # Age vs Spending correlation
    corr_val = df['Age'].corr(df['Total Amount'])
    print(f"  Correlation between Age and Total Amount: {corr_val:.4f}")

    # Age Group vs Product Category preference
    q4_pref = df.groupby(['Age Group', 'Product Category'], observed=False).agg(
        Count=('Transaction ID', 'count')
    ).reset_index()
    # Calculate percentage within age group
    q4_totals = q4_pref.groupby('Age Group', observed=False)['Count'].transform('sum')
    q4_pref['Percentage'] = (q4_pref['Count'] / q4_totals) * 100

    plt.figure(figsize=(10, 6))
    sns.barplot(data=q4_pref, x='Age Group', y='Percentage', hue='Product Category', palette='Set2')
    plt.title('Product Category Preference by Age Group')
    plt.ylabel('Percentage of Purchases (%)')
    plt.xlabel('Age Group')
    plt.tight_layout()
    plt.savefig('assets/plots/q4_age_product_pref.png', dpi=150)
    plt.close()

    # Scatter plot with regression line of Age vs Total Amount
    plt.figure(figsize=(9, 6))
    sns.regplot(data=df, x='Age', y='Total Amount', scatter_kws={'alpha':0.4, 'color':'blue'}, line_kws={'color':'red'})
    plt.title(f'Scatter Plot: Age vs Total Amount (Corr: {corr_val:.4f})')
    plt.xlabel('Age')
    plt.ylabel('Total Amount ($)')
    plt.tight_layout()
    plt.savefig('assets/plots/q4_age_vs_spending_scatter.png', dpi=150)
    plt.close()

    eda_data['q4_correlation'] = {'age_spending_corr': float(corr_val)}
    eda_data['q4_pref'] = q4_pref.to_dict(orient='records')

    # --- Q5: How do customers adapt their shopping habits during seasonal trends? ---
    print("- Question 5...")
    q5_seasonal = df.groupby(['Season', 'Product Category']).agg(
        Total_Sales=('Total Amount', 'sum'),
        Transaction_Count=('Transaction ID', 'count'),
        Avg_Spend=('Total Amount', 'mean')
    ).reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=q5_seasonal, x='Season', y='Total_Sales', hue='Product Category', palette='viridis')
    plt.title('Seasonal Sales by Product Category')
    plt.ylabel('Total Sales ($)')
    plt.xlabel('Season')
    plt.tight_layout()
    plt.savefig('assets/plots/q5_seasonal_sales.png', dpi=150)
    plt.close()

    eda_data['q5_seasonal'] = q5_seasonal.to_dict(orient='records')

    # --- Q6: Are there distinct purchasing behaviors based on the number of items bought per transaction? ---
    print("- Question 6...")
    q6_qty = df.groupby('Quantity').agg(
        Avg_Price_Per_Unit=('Price per Unit', 'mean'),
        Avg_Total_Amount=('Total Amount', 'mean'),
        Transaction_Count=('Transaction ID', 'count')
    ).reset_index()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=q6_qty, x='Quantity', y='Avg_Total_Amount', palette='rocket')
    plt.title('Average Spending by Quantity Purchased')
    plt.ylabel('Average Total Amount ($)')
    plt.xlabel('Quantity of Items')
    plt.tight_layout()
    plt.savefig('assets/plots/q6_qty_spending.png', dpi=150)
    plt.close()

    # Product category share per Quantity
    q6_qty_cat = df.groupby(['Quantity', 'Product Category']).size().unstack(fill_value=0)
    q6_qty_cat_pct = q6_qty_cat.div(q6_qty_cat.sum(axis=1), axis=0) * 100
    q6_qty_cat_pct = q6_qty_cat_pct.reset_index().melt(id_vars='Quantity', value_name='Percentage')

    plt.figure(figsize=(10, 5))
    sns.barplot(data=q6_qty_cat_pct, x='Quantity', y='Percentage', hue='Product Category', palette='Set1')
    plt.title('Product Category Share by Quantity Purchased')
    plt.ylabel('Percentage of Sales (%)')
    plt.xlabel('Quantity')
    plt.tight_layout()
    plt.savefig('assets/plots/q6_qty_product_share.png', dpi=150)
    plt.close()

    eda_data['q6_qty'] = q6_qty.to_dict(orient='records')
    eda_data['q6_qty_category'] = q6_qty_cat_pct.to_dict(orient='records')

    # --- Q7: What insights can be gleaned from the distribution of product prices within each category? ---
    print("- Question 7...")
    # Box plot of Price per Unit by Category
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x='Product Category', y='Price per Unit', palette='Set3')
    plt.title('Distribution of Price per Unit by Product Category')
    plt.ylabel('Price per Unit ($)')
    plt.xlabel('Product Category')
    plt.tight_layout()
    plt.savefig('assets/plots/q7_price_distribution.png', dpi=150)
    plt.close()

    # Violin plot of Total Amount by Category
    plt.figure(figsize=(9, 6))
    sns.violinplot(data=df, x='Product Category', y='Total Amount', palette='Set3')
    plt.title('Distribution of Total Spend by Product Category')
    plt.ylabel('Total Spend ($)')
    plt.xlabel('Product Category')
    plt.tight_layout()
    plt.savefig('assets/plots/q7_total_amount_distribution.png', dpi=150)
    plt.close()

    # Numerical description of price per category
    q7_desc = df.groupby('Product Category')['Price per Unit'].describe().reset_index()
    eda_data['q7_price_stats'] = q7_desc.to_dict(orient='records')

    # General Statistics Summary
    eda_data['general_stats'] = {
        'total_revenue': float(df['Total Amount'].sum()),
        'total_transactions': int(df.shape[0]),
        'average_spend_per_transaction': float(df['Total Amount'].mean()),
        'average_quantity_per_transaction': float(df['Quantity'].mean()),
        'unique_customers': int(df['Customer ID'].nunique()),
        'age_range': [int(df['Age'].min()), int(df['Age'].max())],
        'avg_age': float(df['Age'].mean()),
        'date_range': [df['Date'].min().strftime('%Y-%m-%d'), df['Date'].max().strftime('%Y-%m-%d')]
    }

    # Save to JSON
    with open('assets/data/eda_results.json', 'w') as f:
        json.dump(eda_data, f, indent=4)
        
    print("Success! Data analysis completed, charts saved, and eda_results.json generated.")

if __name__ == '__main__':
    main()
