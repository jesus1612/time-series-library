#!/usr/bin/env python3
"""
Test with Real Cash Flow Dataset
Demonstrates how to use tslib with actual business data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import tslib models
from tslib import ARModel, MAModel, ARMAModel, ARIMAModel

def clean_currency(value):
    """Clean currency string to float"""
    if isinstance(value, str):
        return float(value.replace('$', '').replace(',', '').strip())
    return float(value)

def main():
    print("=" * 70)
    print("TSLib - Real Cash Flow Analysis")
    print("=" * 70)
    
    # 1. Load and prepare data
    print("\n1. Loading and preparing data...")
    csv_path = Path(__file__).parent.parent / "dataset efecitivo mensual.csv"
    
    if not csv_path.exists():
        print(f"   ERROR: Dataset not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} months of data")
    print(f"   Date range: {df['Mes'].iloc[0]} to {df['Mes'].iloc[-1]}")
    
    # Clean currency values
    df['Cash In'] = df['Cash In'].apply(clean_currency)
    df['Cash Out'] = df['Cash Out'].apply(clean_currency)
    
    # Calculate net cash flow
    df['Net Cash Flow'] = df['Cash In'] - df['Cash Out']
    
    print(f"\n   Cash In  - Mean: ${df['Cash In'].mean():,.2f}")
    print(f"   Cash Out - Mean: ${df['Cash Out'].mean():,.2f}")
    print(f"   Net Flow - Mean: ${df['Net Cash Flow'].mean():,.2f}")
    print(f"   Net Flow - Std:  ${df['Net Cash Flow'].std():,.2f}")
    
    # 2. Analyze Cash Out (expenses) with ARIMA
    print("\n" + "=" * 70)
    print("2. Analyzing Cash Out (Expenses) with ARIMA")
    print("=" * 70)
    
    cash_out = df['Cash Out'].values
    
    # Fit ARIMA model with automatic order selection
    print("\n   Fitting ARIMA model...")
    arima_model = ARIMAModel(
        auto_select=True,
        max_p=3,
        max_q=3,
        max_d=2,
        validation=False  # Skip validation for faster results
    )
    arima_model.fit(cash_out)
    
    p, d, q = arima_model.order
    print(f"   Selected model: ARIMA({p}, {d}, {q})")
    
    # Generate forecast
    print("\n   Generating 6-month forecast...")
    forecast_steps = 6
    forecast = arima_model.predict(steps=forecast_steps)
    
    print("\n   Forecasted Cash Out (next 6 months):")
    for i, value in enumerate(forecast, 1):
        print(f"      Month {i}: ${value:,.2f}")
    
    # Evaluate model
    print("\n   Model Diagnostics:")
    residuals = arima_model.get_residuals()
    print(f"      Residuals Mean: ${residuals.mean():,.2f} (should be ≈ 0)")
    print(f"      Residuals Std:  ${residuals.std():,.2f}")
    
    # 3. Analyze Net Cash Flow with different models
    print("\n" + "=" * 70)
    print("3. Comparing Models on Net Cash Flow")
    print("=" * 70)
    
    net_flow = df['Net Cash Flow'].values
    
    # Split data for validation
    train_size = int(0.8 * len(net_flow))
    train_data = net_flow[:train_size]
    test_data = net_flow[train_size:]
    
    print(f"\n   Train size: {train_size} months")
    print(f"   Test size:  {len(test_data)} months")
    
    models_to_test = {
        'AR': ARModel(auto_select=True, max_order=5, validation=False),
        'MA': MAModel(auto_select=True, max_order=5, validation=False),
        'ARMA': ARMAModel(auto_select=True, max_p=3, max_q=3, validation=False),
        'ARIMA': ARIMAModel(auto_select=True, max_p=3, max_q=3, max_d=2, validation=False)
    }
    
    results = {}
    
    for name, model in models_to_test.items():
        print(f"\n   Testing {name} model...")
        try:
            # Fit on training data
            model.fit(train_data)
            
            # Forecast test period
            test_forecast = model.predict(steps=len(test_data))
            
            # Calculate metrics
            mse = np.mean((test_data - test_forecast) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(test_data - test_forecast))
            
            # Get model order
            if hasattr(model, 'order'):
                if isinstance(model.order, tuple):
                    order = f"{model.order}"
                else:
                    order = f"({model.order})"
            else:
                order = "N/A"
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'order': order
            }
            
            print(f"      Order: {order}")
            print(f"      RMSE: ${rmse:,.2f}")
            print(f"      MAE:  ${mae:,.2f}")
            
        except Exception as e:
            print(f"      ERROR: {str(e)}")
            continue
    
    # 4. Best model summary
    print("\n" + "=" * 70)
    print("4. Model Comparison Summary")
    print("=" * 70)
    
    if results:
        best_model_name = min(results.keys(), key=lambda k: results[k]['rmse'])
        
        print("\n   Model Performance (RMSE):")
        for name in sorted(results.keys(), key=lambda k: results[k]['rmse']):
            rmse = results[name]['rmse']
            order = results[name]['order']
            marker = " ← BEST" if name == best_model_name else ""
            print(f"      {name:6s} {order:10s}: ${rmse:,.2f}{marker}")
        
        print(f"\n   Best Model: {best_model_name}")
        best_model = results[best_model_name]['model']
        
        # Generate final forecast with best model
        print("\n   Refitting best model on full dataset...")
        best_model.fit(net_flow)
        final_forecast = best_model.predict(steps=6)
        
        print("\n   Net Cash Flow Forecast (next 6 months):")
        for i, value in enumerate(final_forecast, 1):
            sign = "+" if value >= 0 else "-"
            print(f"      Month {i}: {sign}${abs(value):,.2f}")
    
    # 5. Recommendations
    print("\n" + "=" * 70)
    print("5. Integration Recommendations")
    print("=" * 70)
    print("""
   For your interface project:
   
   1. Installation:
      pip install -e /Users/genaromelgar/escom/TT/time-series-library
   
   2. Import in your code:
      from tslib import ARModel, MAModel, ARMAModel, ARIMAModel
   
   3. Basic usage pattern:
      model = ARIMAModel(auto_select=True)
      model.fit(your_data)
      forecast = model.predict(steps=6)
   
   4. Get results:
      summary = model.summary()
      diagnostics = model.get_residual_diagnostics()
      analysis = model.get_exploratory_analysis()
   
   5. Visualizations:
      model.plot_diagnostics()  # Model validation
      model.plot_forecast(steps=12)  # Forecast visualization
    """)
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()

