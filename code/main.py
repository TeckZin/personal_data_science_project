import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


plt.style.use('default')
sns.set_palette("husl")

def create_data_overview_plots(df):
    """Create overview plots of the dataset"""

    print("\nCreating data overview visualizations...")


    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Stock Data Overview', fontsize=16, fontweight='bold')


    axes[0,0].hist(df['PE_Ratio'].dropna(), bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Distribution of P/E Ratios')
    axes[0,0].set_xlabel('P/E Ratio')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(df['PE_Ratio'].median(), color='red', linestyle='--', label=f'Median: {df["PE_Ratio"].median():.1f}')
    axes[0,0].legend()


    valid_data = df.dropna(subset=['PE_Ratio', 'PEG_Ratio'])
    axes[0,1].scatter(valid_data['PE_Ratio'], valid_data['PEG_Ratio'], alpha=0.7, s=60)
    axes[0,1].set_title('PEG Ratio vs P/E Ratio')
    axes[0,1].set_xlabel('P/E Ratio')
    axes[0,1].set_ylabel('PEG Ratio')
    axes[0,1].grid(True, alpha=0.3)


    for idx, row in valid_data.iterrows():
        axes[0,1].annotate(row['Ticker'], (row['PE_Ratio'], row['PEG_Ratio']),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)


    axes[0,2].boxplot(df['DE_Ratio'].dropna(), patch_artist=True)
    axes[0,2].set_title('Debt-to-Equity Ratio Distribution')
    axes[0,2].set_ylabel('D/E Ratio')
    axes[0,2].grid(True, alpha=0.3)


    cash_flow_data = df.dropna(subset=['Cash_Flow_Yield'])
    axes[1,0].bar(range(len(cash_flow_data)), cash_flow_data['Cash_Flow_Yield'],
                  color='lightgreen', alpha=0.7)
    axes[1,0].set_title('Cash Flow Yield by Stock')
    axes[1,0].set_xlabel('Stock Index')
    axes[1,0].set_ylabel('Cash Flow Yield')
    axes[1,0].set_xticks(range(len(cash_flow_data)))
    axes[1,0].set_xticklabels(cash_flow_data['Ticker'], rotation=45)


    axes[1,1].hist(df['Profit_Growth'].dropna(), bins=10, alpha=0.7, color='orange', edgecolor='black')
    axes[1,1].set_title('Profit Growth Distribution')
    axes[1,1].set_xlabel('Profit Growth (%)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].axvline(0, color='red', linestyle='--', label='Zero Growth')
    axes[1,1].legend()


    if 'EPS_Surprise' in df.columns and df['EPS_Surprise'].notna().sum() > 0:
        surprise_data = df.dropna(subset=['EPS_Surprise'])
        colors = ['red' if x <= 0 else 'green' for x in surprise_data['EPS_Surprise']]
        axes[1,2].bar(range(len(surprise_data)), surprise_data['EPS_Surprise'], color=colors, alpha=0.7)
        axes[1,2].set_title('EPS Surprise by Stock')
        axes[1,2].set_xlabel('Stock Index')
        axes[1,2].set_ylabel('EPS Surprise')
        axes[1,2].set_xticks(range(len(surprise_data)))
        axes[1,2].set_xticklabels(surprise_data['Ticker'], rotation=45)
        axes[1,2].axhline(0, color='black', linestyle='-', alpha=0.5)
    else:
        axes[1,2].text(0.5, 0.5, 'No EPS Surprise\nData Available',
                      ha='center', va='center', transform=axes[1,2].transAxes, fontsize=12)
        axes[1,2].set_title('EPS Surprise Data')

    plt.tight_layout()
    plt.savefig('stock_data_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_correlation_heatmap(df, feature_columns):
    """Create correlation heatmap of features"""

    print("\nCreating correlation heatmap...")


    numeric_cols = feature_columns.copy()
    if 'EPS_Surprise' in df.columns:
        numeric_cols.append('EPS_Surprise')

    correlation_data = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_data, dtype=bool))

    sns.heatmap(correlation_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')

    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_logistic_regression_plots(log_results, df):
    """Create visualizations for logistic regression results"""

    if log_results is None:
        return

    print("\nCreating logistic regression visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Logistic Regression Analysis', fontsize=16, fontweight='bold')


    feature_importance = log_results['feature_importance']
    colors = ['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]

    axes[0,0].barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors, alpha=0.7)
    axes[0,0].set_title('Feature Coefficients (Logistic Regression)')
    axes[0,0].set_xlabel('Coefficient Value')
    axes[0,0].axvline(0, color='black', linestyle='-', alpha=0.5)
    axes[0,0].grid(True, alpha=0.3)


    predictions = log_results['predictions']
    axes[0,1].hist(predictions['Predicted_Prob_Undervalued'], bins=15, alpha=0.7,
                   color='lightblue', edgecolor='black')
    axes[0,1].set_title('Distribution of Predicted Probabilities')
    axes[0,1].set_xlabel('Probability of Being Undervalued')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
    axes[0,1].legend()


    actual_labels = predictions['Actual_Label']
    predicted_probs = predictions['Predicted_Prob_Undervalued']


    undervalued_probs = predicted_probs[actual_labels == 1]
    overvalued_probs = predicted_probs[actual_labels == 0]

    axes[1,0].hist([overvalued_probs, undervalued_probs], bins=10, alpha=0.7,
                   label=['Actually Overvalued', 'Actually Undervalued'],
                   color=['red', 'green'])
    axes[1,0].set_title('Predicted Probabilities by Actual Class')
    axes[1,0].set_xlabel('Predicted Probability of Being Undervalued')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].legend()
    axes[1,0].axvline(0.5, color='black', linestyle='--', alpha=0.5)


    tickers = predictions['Ticker']
    probs = predictions['Predicted_Prob_Undervalued']
    colors = ['green' if p > 0.5 else 'red' for p in probs]

    axes[1,1].bar(range(len(tickers)), probs, color=colors, alpha=0.7)
    axes[1,1].set_title('Undervaluation Probability by Stock')
    axes[1,1].set_xlabel('Stocks')
    axes[1,1].set_ylabel('Probability of Being Undervalued')
    axes[1,1].set_xticks(range(len(tickers)))
    axes[1,1].set_xticklabels(tickers, rotation=45)
    axes[1,1].axhline(0.5, color='black', linestyle='--', alpha=0.5)
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('logistic_regression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_linear_regression_plots(lin_results, df):
    """Create visualizations for linear regression results"""

    if lin_results is None:
        return

    print("\nCreating linear regression visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Linear Regression Analysis', fontsize=16, fontweight='bold')


    feature_coef = lin_results['feature_coefficients']
    colors = ['green' if x > 0 else 'red' for x in feature_coef['Coefficient']]

    axes[0,0].barh(feature_coef['Feature'], feature_coef['Coefficient'], color=colors, alpha=0.7)
    axes[0,0].set_title('Feature Coefficients (Linear Regression)')
    axes[0,0].set_xlabel('Coefficient Value')
    axes[0,0].axvline(0, color='black', linestyle='-', alpha=0.5)
    axes[0,0].grid(True, alpha=0.3)


    predictions = lin_results['predictions']
    target_col = lin_results['target_variable']

    actual = predictions[f'Actual_{target_col}']
    predicted = predictions[f'Predicted_{target_col}']

    axes[0,1].scatter(actual, predicted, alpha=0.7, s=60)
    axes[0,1].plot([actual.min(), actual.max()], [actual.min(), actual.max()],
                   'r--', lw=2, label='Perfect Prediction')
    axes[0,1].set_title(f'Actual vs Predicted {target_col}')
    axes[0,1].set_xlabel(f'Actual {target_col}')
    axes[0,1].set_ylabel(f'Predicted {target_col}')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)


    r2 = lin_results['r2_score']
    axes[0,1].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0,1].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


    residuals = predictions['Prediction_Error']
    axes[1,0].scatter(predicted, residuals, alpha=0.7, s=60)
    axes[1,0].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes[1,0].set_title('Residuals vs Predicted Values')
    axes[1,0].set_xlabel(f'Predicted {target_col}')
    axes[1,0].set_ylabel('Residuals')
    axes[1,0].grid(True, alpha=0.3)


    tickers = predictions['Ticker']
    actual_values = actual
    predicted_values = predicted

    x_pos = np.arange(len(tickers))
    width = 0.35

    axes[1,1].bar(x_pos - width/2, actual_values, width, label='Actual', alpha=0.7, color='blue')
    axes[1,1].bar(x_pos + width/2, predicted_values, width, label='Predicted', alpha=0.7, color='orange')
    axes[1,1].set_title(f'{target_col} by Stock: Actual vs Predicted')
    axes[1,1].set_xlabel('Stocks')
    axes[1,1].set_ylabel(target_col)
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(tickers, rotation=45)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('linear_regression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_model_comparison_plot(log_results, lin_results):
    """Create model comparison visualization"""

    print("\nCreating model comparison visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')


    if log_results and lin_results:
        log_importance = log_results['feature_importance'].set_index('Feature')['Abs_Coefficient']
        lin_importance = lin_results['feature_coefficients'].set_index('Feature')['Abs_Coefficient']


        log_importance_norm = log_importance / log_importance.max()
        lin_importance_norm = lin_importance / lin_importance.max()

        comparison_df = pd.DataFrame({
            'Logistic': log_importance_norm,
            'Linear': lin_importance_norm
        }).fillna(0)

        comparison_df.plot(kind='bar', ax=axes[0], alpha=0.7, width=0.8)
        axes[0].set_title('Normalized Feature Importance Comparison')
        axes[0].set_xlabel('Features')
        axes[0].set_ylabel('Normalized Importance')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)


    if log_results and lin_results:
        metrics = ['Accuracy/R²', 'Model Complexity']
        logistic_scores = [log_results['accuracy'], 0.7]
        linear_scores = [lin_results['r2_score'], 0.8]

        x = np.arange(len(metrics))
        width = 0.35

        axes[1].bar(x - width/2, logistic_scores, width, label='Logistic Regression', alpha=0.7, color='skyblue')
        axes[1].bar(x + width/2, linear_scores, width, label='Linear Regression', alpha=0.7, color='lightcoral')

        axes[1].set_title('Model Performance Metrics')
        axes[1].set_ylabel('Score')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(metrics)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)


        for i, (log_val, lin_val) in enumerate(zip(logistic_scores, linear_scores)):
            axes[1].text(i - width/2, log_val + 0.01, f'{log_val:.3f}', ha='center', va='bottom')
            axes[1].text(i + width/2, lin_val + 0.01, f'{lin_val:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_investment_insights_plot(log_results, df):
    """Create investment insights visualization"""

    if log_results is None:
        return

    print("\nCreating investment insights visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Investment Insights Dashboard', fontsize=16, fontweight='bold')

    predictions = log_results['predictions']


    pe_data = df.set_index('Ticker')['PE_Ratio']
    prob_data = predictions.set_index('Ticker')['Predicted_Prob_Undervalued']


    risk_return = pd.DataFrame({'PE_Ratio': pe_data, 'Undervalued_Prob': prob_data}).dropna()

    colors = ['green' if p > 0.5 else 'red' for p in risk_return['Undervalued_Prob']]
    scatter = axes[0,0].scatter(risk_return['PE_Ratio'], risk_return['Undervalued_Prob'],
                               c=colors, alpha=0.7, s=100)

    axes[0,0].set_title('Risk-Return Matrix')
    axes[0,0].set_xlabel('P/E Ratio (Risk Proxy)')
    axes[0,0].set_ylabel('Probability of Being Undervalued')
    axes[0,0].axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Decision Threshold')
    axes[0,0].axvline(risk_return['PE_Ratio'].median(), color='blue', linestyle='--', alpha=0.5, label='Median P/E')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()


    for ticker, row in risk_return.iterrows():
        axes[0,0].annotate(ticker, (row['PE_Ratio'], row['Undervalued_Prob']),
                          xytext=(5, 5), textcoords='offset points', fontsize=9)


    high_prob_stocks = predictions[predictions['Predicted_Prob_Undervalued'] > 0.7]['Ticker']
    medium_prob_stocks = predictions[(predictions['Predicted_Prob_Undervalued'] > 0.3) &
                                   (predictions['Predicted_Prob_Undervalued'] <= 0.7)]['Ticker']
    low_prob_stocks = predictions[predictions['Predicted_Prob_Undervalued'] <= 0.3]['Ticker']

    categories = ['Strong Buy\n(>70%)', 'Hold\n(30-70%)', 'Avoid\n(<30%)']
    counts = [len(high_prob_stocks), len(medium_prob_stocks), len(low_prob_stocks)]
    colors_pie = ['green', 'yellow', 'red']

    axes[0,1].pie(counts, labels=categories, colors=colors_pie, autopct='%1.0f%%', startangle=90)
    axes[0,1].set_title('Investment Recommendations')


    top_stocks = predictions.nlargest(5, 'Predicted_Prob_Undervalued')
    axes[1,0].barh(top_stocks['Ticker'], top_stocks['Predicted_Prob_Undervalued'],
                   color='green', alpha=0.7)
    axes[1,0].set_title('Top 5 Most Likely Undervalued Stocks')
    axes[1,0].set_xlabel('Probability of Being Undervalued')
    axes[1,0].grid(True, alpha=0.3)


    for i, (ticker, prob) in enumerate(zip(top_stocks['Ticker'], top_stocks['Predicted_Prob_Undervalued'])):
        axes[1,0].text(prob + 0.01, i, f'{prob:.1%}', va='center')


    if all(col in df.columns for col in ['DE_Ratio', 'Cash_Flow_Yield']):
        health_data = df.dropna(subset=['DE_Ratio', 'Cash_Flow_Yield'])


        health_data['Health_Score'] = (1 / (1 + health_data['DE_Ratio'])) * 0.5 + \
                                     np.clip(health_data['Cash_Flow_Yield'] * 10, 0, 0.5)

        axes[1,1].bar(health_data['Ticker'], health_data['Health_Score'],
                     color='lightblue', alpha=0.7)
        axes[1,1].set_title('Financial Health Score')
        axes[1,1].set_xlabel('Stocks')
        axes[1,1].set_ylabel('Health Score (0-1)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'Insufficient Data\nfor Health Score',
                      ha='center', va='center', transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].set_title('Financial Health Score')

    plt.tight_layout()
    plt.savefig('investment_insights.png', dpi=300, bbox_inches='tight')
    plt.show()

def load_and_prepare_data(csv_file='ml_stock_data.csv'):
    """Load and prepare the stock data for ML models"""

    print("Loading and preparing stock data...")
    print("=" * 50)

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} stocks with {df.shape[1]} features")

    print(f"\nData Overview:")
    print(df.head())

    print(f"\nMissing Values Check:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"  {col}: {missing}/{len(df)} missing")

    feature_columns = ['PE_Ratio', 'PEG_Ratio', 'DE_Ratio', 'PS_Ratio',
                      'Profit_Growth', 'Cash_Flow_Yield']

    print(f"\nEPS Surprise Analysis:")
    eps_available = df['EPS_Surprise'].notna().sum()
    print(f"Stocks with EPS surprise data: {eps_available}/{len(df)}")

    if eps_available == 0:
        print("No EPS surprise data available. Using alternative target variable.")
        median_pe = df['PE_Ratio'].median()
        df['Undervalued'] = (df['PE_Ratio'] < median_pe).astype(int)
        target_description = f"stocks with P/E < median P/E ({median_pe:.2f})"
        df_clean = df.dropna(subset=feature_columns)
    else:
        print(f"EPS Surprise range: {df['EPS_Surprise'].min():.3f} to {df['EPS_Surprise'].max():.3f}")

        positive_surprises = (df['EPS_Surprise'] > 0).sum()
        negative_surprises = (df['EPS_Surprise'] <= 0).sum()

        print(f"Positive surprises: {positive_surprises}")
        print(f"Negative/zero surprises: {negative_surprises}")

        if positive_surprises == 0 or negative_surprises == 0:
            print("All EPS surprises have the same sign. Using median-based classification.")
            median_surprise = df['EPS_Surprise'].median()
            df['Undervalued'] = (df['EPS_Surprise'] > median_surprise).astype(int)
            target_description = f"stocks with EPS surprise > median ({median_surprise:.3f})"
        else:
            df['Undervalued'] = (df['EPS_Surprise'] > 0).astype(int)
            target_description = "stocks with positive EPS surprise"

        df_clean = df.dropna(subset=['EPS_Surprise'] + feature_columns)

    print(f"\nAfter cleaning: {len(df_clean)} stocks ready for analysis")

    class_distribution = df_clean['Undervalued'].value_counts()
    print(f"Classification target ({target_description}):")
    print(f"  Class 0 (overvalued): {class_distribution.get(0, 0)} stocks")
    print(f"  Class 1 (undervalued): {class_distribution.get(1, 0)} stocks")

    if len(class_distribution) < 2:
        print("WARNING: Only one class found. Adjusting classification threshold...")
        if 'EPS_Surprise' in df_clean.columns:
            threshold = df_clean['EPS_Surprise'].quantile(0.6)
            df_clean['Undervalued'] = (df_clean['EPS_Surprise'] > threshold).astype(int)
        else:
            threshold = df_clean['PE_Ratio'].quantile(0.4)
            df_clean['Undervalued'] = (df_clean['PE_Ratio'] < threshold).astype(int)

        class_distribution = df_clean['Undervalued'].value_counts()
        print(f"Adjusted classification:")
        print(f"  Class 0: {class_distribution.get(0, 0)} stocks")
        print(f"  Class 1: {class_distribution.get(1, 0)} stocks")

    return df_clean, feature_columns

def logistic_regression_analyze(df, feature_columns):
    """Logistic regression with enhanced error handling"""

    print("\nLOGISTIC REGRESSION ANALYSIS")
    print("=" * 60)
    print("Purpose: Classify stocks as overvalued (0) or undervalued (1)")
    print("Target: Based on earnings surprise (beat expectations = undervalued)")

    X = df[feature_columns]
    y = df['Undervalued']

    print(f"\nFeatures used: {feature_columns}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    X = X.fillna(X.median())

    if len(y.unique()) < 2:
        print("Error: Only one class available. Cannot perform classification.")
        return None

    if len(y.unique()) == 2 and min(y.value_counts()) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(random_state=42, max_iter=1000)
    log_model.fit(X_train_scaled, y_train)

    y_pred = log_model.predict(X_test_scaled)
    y_pred_proba = log_model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nLOGISTIC REGRESSION RESULTS:")
    print("=" * 40)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Overvalued', 'Undervalued']))

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print("Predicted:  Overvalued  Undervalued")
    print(f"Overvalued:      {cm[0,0]}          {cm[0,1]}")
    print(f"Undervalued:     {cm[1,0]}          {cm[1,1]}")

    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': log_model.coef_[0],
        'Abs_Coefficient': np.abs(log_model.coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)

    print(f"\nFeature Importance (Coefficients):")
    print(feature_importance.round(4))

    all_predictions = pd.DataFrame({
        'Ticker': df['Ticker'],
        'Actual_Surprise': df['EPS_Surprise'] if 'EPS_Surprise' in df.columns else None,
        'Actual_Label': df['Undervalued'],
        'Predicted_Prob_Undervalued': log_model.predict_proba(scaler.transform(X.fillna(X.median())))[:, 1],
        'Predicted_Label': log_model.predict(scaler.transform(X.fillna(X.median())))
    })

    print(f"\nStock Predictions:")
    print(all_predictions.round(3))

    return {
        'model': log_model,
        'scaler': scaler,
        'accuracy': accuracy,
        'feature_importance': feature_importance,
        'predictions': all_predictions,
        'test_data': (X_test, y_test, y_pred, y_pred_proba)
    }

def linear_regression_analyze(df, feature_columns):
    """Linear regression with enhanced error handling"""

    print("\nLINEAR REGRESSION ANALYSIS")
    print("=" * 60)

    if 'EPS_Surprise' in df.columns and df['EPS_Surprise'].notna().sum() > 0:
        target_col = 'EPS_Surprise'
        print("Purpose: Predict EPS surprise as a continuous variable")
        print("Target: EPS_Surprise = Actual_EPS - Expected_EPS")
    else:
        median_pe = df['PE_Ratio'].median()
        df['PE_Relative'] = df['PE_Ratio'] - median_pe
        target_col = 'PE_Relative'
        print("Purpose: Predict relative P/E performance as a continuous variable")
        print("Target: PE_Relative = Stock_PE - Median_PE")

    X = df[feature_columns]
    y = df[target_col]

    print(f"\nFeatures used: {feature_columns}")
    print(f"Target variable: {target_col}")
    print(f"Target range: {y.min():.3f} to {y.max():.3f}")
    print(f"Target mean: {y.mean():.3f}, std: {y.std():.3f}")

    X = X.fillna(X.median())
    y = y.fillna(y.median())

    valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[valid_indices]
    y = y[valid_indices]

    if len(X) < 4:
        print("Error: Insufficient data for linear regression analysis.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lin_model = LinearRegression()
    lin_model.fit(X_train_scaled, y_train)

    y_pred_train = lin_model.predict(X_train_scaled)
    y_pred_test = lin_model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f"\nLINEAR REGRESSION RESULTS:")
    print("=" * 40)
    print(f"Training R²: {train_r2:.3f}")
    print(f"Testing R²: {test_r2:.3f}")
    print(f"Training RMSE: {train_rmse:.3f}")
    print(f"Testing RMSE: {test_rmse:.3f}")
    print(f"Testing MAE: {test_mae:.3f}")

    feature_coef = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': lin_model.coef_,
        'Abs_Coefficient': np.abs(lin_model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)

    print(f"\nFeature Coefficients:")
    print(f"Intercept: {lin_model.intercept_:.4f}")
    print(feature_coef.round(4))

    all_X_scaled = scaler.transform(X.fillna(X.median()))
    all_predictions = pd.DataFrame({
        'Ticker': df.loc[valid_indices, 'Ticker'],
        f'Actual_{target_col}': y,
        f'Predicted_{target_col}': lin_model.predict(all_X_scaled),
        'Prediction_Error': y - lin_model.predict(all_X_scaled)
    })

    print(f"\n{target_col} Predictions:")
    print(all_predictions.round(4))

    return {
        'model': lin_model,
        'scaler': scaler,
        'r2_score': test_r2,
        'rmse': test_rmse,
        'mae': test_mae,
        'feature_coefficients': feature_coef,
        'predictions': all_predictions,
        'test_data': (X_test, y_test, y_pred_test),
        'target_variable': target_col
    }

def main_analysis_with_visualizations():
    """Main function with comprehensive visualizations"""

    print("STOCK VALUATION ANALYSIS WITH DATA VISUALIZATIONS")
    print("=" * 80)
    print("Methodology: Using financial ratios to predict stock valuation")
    print("Outputs: Analysis results + PNG image files for each visualization")


    df, feature_columns = load_and_prepare_data()

    if len(df) < 5:
        print("Insufficient data for analysis. Need at least 5 stocks with complete data.")
        return


    create_data_overview_plots(df)
    create_correlation_heatmap(df, feature_columns)


    log_results = logistic_regression_analyze(df, feature_columns)
    lin_results = linear_regression_analyze(df, feature_columns)


    if log_results:
        create_logistic_regression_plots(log_results, df)

    if lin_results:
        create_linear_regression_plots(lin_results, df)


    create_model_comparison_plot(log_results, lin_results)

    if log_results:
        create_investment_insights_plot(log_results, df)


    print(f"\nVISUALIZATION SUMMARY")
    print("=" * 40)
    print("Generated visualization files:")
    print("1. stock_data_overview.png - Data distribution and overview")
    print("2. correlation_heatmap.png - Feature correlation matrix")
    if log_results:
        print("3. logistic_regression_analysis.png - Classification results")
    if lin_results:
        print("4. linear_regression_analysis.png - Regression results")
    print("5. model_comparison.png - Performance comparison")
    if log_results:
        print("6. investment_insights.png - Investment recommendations")

    print(f"\nMODEL COMPARISON SUMMARY")
    print("=" * 40)
    if log_results:
        print(f"Logistic Regression Accuracy: {log_results['accuracy']:.3f}")
    if lin_results:
        print(f"Linear Regression R²: {lin_results['r2_score']:.3f}")

    print(f"\nKEY INSIGHTS:")
    print("=" * 20)
    if log_results:
        top_feature = log_results['feature_importance'].iloc[0]
        print(f"Most important feature (Classification): {top_feature['Feature']}")

        best_stock = log_results['predictions'].loc[
            log_results['predictions']['Predicted_Prob_Undervalued'].idxmax()
        ]
        print(f"Most likely undervalued stock: {best_stock['Ticker']} ({best_stock['Predicted_Prob_Undervalued']:.1%})")

    return log_results, lin_results


if __name__ == "__main__":
    logistic_results, linear_results = main_analysis_with_visualizations()
