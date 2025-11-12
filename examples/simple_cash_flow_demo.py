#!/usr/bin/env python3
"""
Simple Cash Flow Demo
Quick demonstration of TSLib with real cash flow data
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import TSLib
from tslib import ARIMAModel

def clean_currency(value):
    """Clean currency string to float"""
    if isinstance(value, str):
        return float(value.replace('$', '').replace(',', '').strip())
    return float(value)

def main():
    print("\n" + "="*60)
    print("TSLib - Simple Cash Flow Demo")
    print("="*60)
    
    # Load data
    csv_path = Path(__file__).parent.parent / "dataset efecitivo mensual.csv"
    df = pd.read_csv(csv_path)
    
    # Clean data
    df['Cash Out'] = df['Cash Out'].apply(clean_currency)
    df['Cash In'] = df['Cash In'].apply(clean_currency)
    df['Net Flow'] = df['Cash In'] - df['Cash Out']
    
    print(f"\n📊 Datos cargados: {len(df)} meses")
    print(f"   Rango: {df['Mes'].iloc[0]} a {df['Mes'].iloc[-1]}")
    
    # Analyze Cash Out
    print("\n💰 Analizando Cash Out (Egresos)...")
    cash_out = df['Cash Out'].values
    
    # Create and fit model
    model = ARIMAModel(auto_select=True, validation=False)
    print("   Ajustando modelo ARIMA...")
    model.fit(cash_out)
    
    # Get model order
    p, d, q = model.order
    print(f"   ✓ Modelo seleccionado: ARIMA({p}, {d}, {q})")
    
    # Generate forecast
    print("\n🔮 Pronóstico para los próximos 6 meses:")
    forecast = model.predict(steps=6)
    
    for i, value in enumerate(forecast, 1):
        print(f"   Mes {i}: ${value:,.2f}")
    
    # Model diagnostics
    residuals = model.get_residuals()
    print(f"\n📈 Diagnósticos del modelo:")
    print(f"   Residuales - Media: ${residuals.mean():,.2f} (ideal: ~0)")
    print(f"   Residuales - Desv. Est: ${residuals.std():,.2f}")
    
    # Compare with last month
    last_month = cash_out[-1]
    first_forecast = forecast[0]
    change = ((first_forecast - last_month) / last_month) * 100
    
    print(f"\n📊 Comparación:")
    print(f"   Último mes real: ${last_month:,.2f}")
    print(f"   Próximo mes (pronóstico): ${first_forecast:,.2f}")
    print(f"   Cambio esperado: {change:+.2f}%")
    
    print("\n" + "="*60)
    print("✅ Demo completada exitosamente!")
    print("="*60)
    
    print("\n💡 Para usar en tu proyecto:")
    print("   1. Instala: pip install -e /path/to/time-series-library")
    print("   2. Importa: from tslib import ARIMAModel")
    print("   3. Usa: model = ARIMAModel(auto_select=True)")
    print("   4. Documenta: Ver docs/INTEGRATION_GUIDE.md\n")

if __name__ == "__main__":
    main()

