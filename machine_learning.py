import os
import json
import kagglehub
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics for plots
sns.set_theme(style="whitegrid")

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
    print("Step 1: Loading retail sales dataset for ML...")
    dataset_dir = kagglehub.dataset_download("mohammadtalib786/retail-sales-dataset")
    csv_path = os.path.join(dataset_dir, "retail_sales_dataset.csv")
    df = pd.read_csv(csv_path)

    # Preprocessing
    df.columns = df.columns.str.strip()
    df['Gender'] = df['Gender'].str.strip()
    df['Product Category'] = df['Product Category'].str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Season'] = df['Month'].apply(get_season)

    # Directories
    os.makedirs('assets/plots', exist_ok=True)
    os.makedirs('assets/data', exist_ok=True)

    ml_data = {}

    # --- Part 1: Customer Segmentation (K-Means Clustering) ---
    print("\nStep 2: Customer Segmentation (K-Means)...")
    # Features for clustering
    cluster_features = ['Age', 'Total Amount', 'Quantity']
    X_clust = df[cluster_features].copy()

    # Scale the features
    scaler = StandardScaler()
    X_clust_scaled = scaler.fit_transform(X_clust)

    # Use n_clusters=4 for rich customer profiles
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_clust_scaled)

    # Profile the clusters
    cluster_profiles = []
    # Let's map cluster labels to meaningful names based on centroids
    centroids = kmeans.cluster_centers_
    # Descale centroids for profiling
    centroids_orig = scaler.inverse_transform(centroids)

    print("Centroids (Age, Total Spend, Quantity):")
    for i in range(n_clusters):
        c = centroids_orig[i]
        size = int(np.sum(df['Cluster'] == i))
        pct = float(size / len(df) * 100)
        print(f"Cluster {i}: Age={c[0]:.1f}, Spend=${c[1]:.1f}, Qty={c[2]:.2f} (Size={size}, Pct={pct:.1f}%)")
        
        # Determine a profile name based on spend and quantity
        if c[1] > 250 and c[2] >= 3:
            name = "High-Volume Spenders"
            desc = "Customers who buy multiple items in a single transaction, leading to high transaction totals. Typically high-value customers."
        elif c[1] > 250 and c[2] < 3:
            name = "Premium Value Buyers"
            desc = "Customers who purchase higher-priced items, resulting in high spend despite buying few items."
        elif c[0] < 35:
            name = "Younger Moderate Shoppers"
            desc = "Younger customers buying basic items, showing moderate spending habits and transaction sizes."
        else:
            name = "Mature Budget Shoppers"
            desc = "Older customers who purchase single items with lower average unit prices, showing highly budget-conscious behavior."
            
        cluster_profiles.append({
            'cluster_id': i,
            'name': name,
            'description': desc,
            'avg_age': float(c[0]),
            'avg_spend': float(c[1]),
            'avg_quantity': float(c[2]),
            'size': size,
            'percentage': pct
        })

    # Save 3D scatter plot of clusters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate dynamic colors from colormap to support any number of clusters without IndexErrors
    cmap = plt.colormaps.get_cmap('tab10')
    colors = [cmap(i) for i in range(n_clusters)]
    
    for i in range(n_clusters):
        clust_df = df[df['Cluster'] == i]
        ax.scatter(clust_df['Age'], clust_df['Total Amount'], clust_df['Quantity'], 
                   color=colors[i], label=cluster_profiles[i]['name'], s=40, alpha=0.7)
                   
    ax.set_xlabel('Age')
    ax.set_ylabel('Total Amount ($)')
    ax.set_zlabel('Quantity')
    ax.set_title('Customer Segments 3D Scatter Plot')
    plt.legend()
    plt.tight_layout()
    plt.savefig('assets/plots/ml_customer_segments.png', dpi=150)
    plt.close()

    # Save Clustering parameters
    ml_data['clustering'] = {
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'centroids': centroids.tolist(),
        'cluster_profiles': cluster_profiles
    }

    # --- Part 2: Regression Modeling (Quantity & Total Amount Predictor) ---
    print("\nStep 3: Preparing Regression modeling...")
    
    # Feature engineering for regression
    # Columns to use: Age, Gender, Product Category, Month, Year, DayOfWeek
    # We will one-hot encode categorical features: Gender, Product Category, DayOfWeek
    
    # We create a mapping of expected categorical values to ensure the JS frontend knows how to format features
    categorical_info = {
        'Gender': sorted(df['Gender'].unique().tolist()),
        'Product_Category': sorted(df['Product Category'].unique().tolist()),
        'DayOfWeek': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    }
    
    # Manual dummy creation to have standard feature orders we can easily recreate in Javascript
    X_features = []
    
    # Numerical features
    X = pd.DataFrame()
    X['Age'] = df['Age']
    X['Month'] = df['Month']
    X['Year'] = df['Year']
    
    # Add dummy variables manually to control naming and order strictly
    for val in categorical_info['Gender']:
        X[f'Gender_{val}'] = (df['Gender'] == val).astype(int)
        
    for val in categorical_info['Product_Category']:
        X[f'Product_Category_{val}'] = (df['Product Category'] == val).astype(int)
        
    for val in categorical_info['DayOfWeek']:
        X[f'DayOfWeek_{val}'] = (df['DayOfWeek'] == val).astype(int)

    feature_cols = X.columns.tolist()
    print("Feature columns constructed for regression model:", feature_cols)
    
    # Targets
    y_qty = df['Quantity']
    y_amt = df['Total Amount']
    
    # Split
    X_train, X_test, y_qty_train, y_qty_test, y_amt_train, y_amt_test = train_test_split(
        X, y_qty, y_amt, test_size=0.2, random_state=42
    )

    # Train Ridge Regression for client-side easy inference
    print("\nTraining Ridge Regression models...")
    model_qty = Ridge(alpha=1.0)
    model_qty.fit(X_train, y_qty_train)
    y_qty_pred = model_qty.predict(X_test)
    
    model_amt = Ridge(alpha=10.0) # slightly higher regularization for spend
    model_amt.fit(X_train, y_amt_train)
    y_amt_pred = model_amt.predict(X_test)

    # Metrics for Ridge
    qty_r2 = r2_score(y_qty_test, y_qty_pred)
    qty_mae = mean_absolute_error(y_qty_test, y_qty_pred)
    amt_r2 = r2_score(y_amt_test, y_amt_pred)
    amt_mae = mean_absolute_error(y_amt_test, y_amt_pred)
    
    print(f"Ridge Quantity Predictor: R2={qty_r2:.4f}, MAE={qty_mae:.4f}")
    print(f"Ridge Spend Predictor: R2={amt_r2:.4f}, MAE={amt_mae:.4f}")

    # Train Random Forest Regressor for metric comparison (shows capacity of more complex models)
    print("\nTraining Random Forest Regressor models for comparison...")
    rf_qty = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf_qty.fit(X_train, y_qty_train)
    rf_qty_pred = rf_qty.predict(X_test)
    
    rf_amt = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf_amt.fit(X_train, y_amt_train)
    rf_amt_pred = rf_amt.predict(X_test)
    
    rf_qty_r2 = r2_score(y_qty_test, rf_qty_pred)
    rf_qty_mae = mean_absolute_error(y_qty_test, rf_qty_pred)
    rf_amt_r2 = r2_score(y_amt_test, rf_amt_pred)
    rf_amt_mae = mean_absolute_error(y_amt_test, rf_amt_pred)
    
    print(f"RF Quantity Predictor: R2={rf_qty_r2:.4f}, MAE={rf_qty_mae:.4f}")
    print(f"RF Spend Predictor: R2={rf_amt_r2:.4f}, MAE={rf_amt_mae:.4f}")

    # Export Regression Parameters
    ml_data['regression'] = {
        'feature_names': feature_cols,
        'categorical_info': categorical_info,
        'quantity_model': {
            'intercept': float(model_qty.intercept_),
            'coefficients': dict(zip(feature_cols, model_qty.coef_.tolist())),
            'r2': float(qty_r2),
            'mae': float(qty_mae),
            'mse': float(mean_squared_error(y_qty_test, y_qty_pred))
        },
        'total_amount_model': {
            'intercept': float(model_amt.intercept_),
            'coefficients': dict(zip(feature_cols, model_amt.coef_.tolist())),
            'r2': float(amt_r2),
            'mae': float(amt_mae),
            'mse': float(mean_squared_error(y_amt_test, y_amt_pred))
        },
        'rf_comparison': {
            'quantity_r2': float(rf_qty_r2),
            'quantity_mae': float(rf_qty_mae),
            'total_amount_r2': float(rf_amt_r2),
            'total_amount_mae': float(rf_amt_mae)
        }
    }

    # Save to JSON
    with open('assets/data/ml_results.json', 'w') as f:
        json.dump(ml_data, f, indent=4)
        
    # Generate data.js for CORS-free offline client-side loading
    try:
        eda_path = 'assets/data/eda_results.json'
        if os.path.exists(eda_path):
            with open(eda_path, 'r') as f_eda:
                eda_json_data = json.load(f_eda)
            
            with open('assets/data/data.js', 'w') as f_js:
                f_js.write("// Autogenerated retail data file for CORS-free loading\n")
                f_js.write("window.edaData = ")
                json.dump(eda_json_data, f_js, indent=4)
                f_js.write(";\n\n")
                f_js.write("window.mlData = ")
                json.dump(ml_data, f_js, indent=4)
                f_js.write(";\n")
            print("Successfully generated assets/data/data.js for offline use.")
        else:
            print("Warning: eda_results.json not found, skipping data.js generation.")
    except Exception as e:
        print(f"Error generating data.js: {e}")
        
    print("\nSuccess! Machine learning models trained, segment plot saved, and ml_results.json generated.")

if __name__ == '__main__':
    main()
