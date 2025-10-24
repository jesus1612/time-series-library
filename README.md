# TSLib - Time Series Library

Una librería completa de análisis de series de tiempo con implementación de ARIMA desde cero, diseñada con principios de programación orientada a objetos y soporte opcional para procesamiento distribuido con PySpark.

## Características

- ✅ **Implementación ARIMA desde cero**: Algoritmos implementados matemáticamente sin dependencias de librerías externas de modelos
- ✅ **Arquitectura orientada a objetos**: Diseño limpio siguiendo principios SOLID
- ✅ **Soporte PySpark opcional**: Procesamiento paralelo de múltiples series con Pandas UDF
- ✅ **API intuitiva**: Balance entre simplicidad y control
- ✅ **Tests comprehensivos**: Alta cobertura de código con unit e integration tests

## Instalación

### Instalación básica

```bash
pip install -r requirements.txt
pip install -e .
```

### Instalación con soporte PySpark

```bash
pip install -r requirements.txt
pip install -r requirements-spark.txt
pip install -e .
```

## Uso Rápido

### ARIMA básico

```python
from tslib.models import ARIMAModel
import pandas as pd

# Load your time series data
data = pd.read_csv("data.csv")["value"]

# Create and fit ARIMA model
model = ARIMAModel(order=(1, 1, 1))
model.fit(data)

# Make predictions
forecast = model.predict(steps=10)
print(forecast)

# View model summary
model.summary()
```

### ARIMA con PySpark (procesamiento paralelo)

```python
from pyspark.sql import SparkSession
from tslib.spark import fit_predict_arima_udf

spark = SparkSession.builder.appName("TimeSeriesAnalysis").getOrCreate()

# DataFrame with multiple time series grouped by 'group_id'
df = spark.read.parquet("multiple_series.parquet")

# Apply ARIMA to each group in parallel
result = df.groupBy("group_id").agg(
    fit_predict_arima_udf("value", order=(1, 1, 1)).alias("forecast")
)
```

## Estructura del Proyecto

```
tslib/
├── core/              # Core algorithms (ARIMA, ACF/PACF, optimization)
├── models/            # High-level model interfaces
├── preprocessing/     # Data transformations and validation
├── metrics/           # Model evaluation metrics
├── spark/             # PySpark integration
└── utils/             # Utility functions
```

## Componentes Matemáticos

### ARIMA(p, d, q)

- **AR (AutoRegressive)**: y_t = c + Σφ_i * y_{t-i} + ε_t
- **MA (Moving Average)**: y_t = μ + ε_t + Σθ_i * ε_{t-i}
- **I (Integration)**: Diferenciación de orden d

### Tests de Estacionariedad

- Augmented Dickey-Fuller (ADF) test
- KPSS test

### Métricas de Evaluación

- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- RMSE, MAE, MAPE

## Desarrollo

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=tslib --cov-report=html

# Format code
black tslib/

# Lint code
flake8 tslib/
```

## Documentación

Ver la carpeta `docs/` para documentación completa:

- [Mathematical Foundations](docs/mathematical_foundations.md)
- [API Reference](docs/api_reference.md)
- [Tutorials](docs/tutorials.md)

## Licencia

MIT License

## Autores

- Genaro Melgar - ESCOM, Instituto Politécnico Nacional


