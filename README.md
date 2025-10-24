# TSLib - Time Series Library

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](https://github.com/genaromelgar/time-series-library)

Una librería completa de análisis de series de tiempo con implementación de ARIMA desde cero, diseñada con principios de programación orientada a objetos y soporte opcional para procesamiento distribuido con PySpark.

## Características

- ✅ **Implementación ARIMA desde cero**: Algoritmos implementados matemáticamente sin dependencias de librerías externas de modelos
- ✅ **Arquitectura orientada a objetos**: Diseño limpio siguiendo principios SOLID con jerarquía de clases bien definida
- ✅ **Paralelización interna automática**: Optimización MLE, ACF/PACF y operaciones intensivas paralelizadas automáticamente
- ✅ **Soporte PySpark opcional**: Procesamiento paralelo de múltiples series con Pandas UDF
- ✅ **API intuitiva**: Balance entre simplicidad y control siguiendo convenciones de scikit-learn
- ✅ **Tests comprehensivos**: Alta cobertura de código con unit, integration y performance tests
- ✅ **Benchmarks de rendimiento**: Comparación detallada entre implementación normal y Spark
- ✅ **Compatibilidad amplia**: Soporte para Python 3.9-3.12

## Requisitos y Compatibilidad

- **Python**: 3.9, 3.10, 3.11, 3.12
- **Java**: 17+ (requerido para funcionalidad PySpark)
- **Dependencias principales**: NumPy ≥1.24.0, SciPy ≥1.10.0, Pandas ≥1.5.0, Matplotlib ≥3.6.0
- **PySpark**: 4.0.1 (opcional, para procesamiento distribuido)
- **Sistema operativo**: Windows, macOS, Linux

### Instalación de Java 17+

**macOS (con Homebrew):**
```bash
brew install openjdk@17
export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"
export JAVA_HOME="/opt/homebrew/opt/openjdk@17"
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install openjdk-17-jdk
export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
```

**Windows:**
1. Descargar desde [Adoptium](https://adoptium.net/temurin/releases/?version=17)
2. Ejecutar el instalador
3. Configurar variable de entorno `JAVA_HOME`

**Verificar instalación:**
```bash
java -version  # Debe mostrar Java 17 o superior
```

## Instalación

### Usando Makefile (Recomendado)

```bash
# Verificar requisitos
make check-version  # Verifica Python 3.9+
make check-java     # Verifica Java 17+ (para Spark)

# Instalación básica
make install

# Con soporte PySpark (requiere Java 17+)
make install-spark

# Con herramientas de desarrollo
make install-dev

# Instalar Java 17+ (si es necesario)
make install-java-macos    # macOS
make install-java-linux    # Linux
make install-java-windows  # Windows

# Ver todas las opciones
make help
```

### Instalación manual

```bash
# Instalación básica
pip install -r requirements.txt
pip install -e .

# Con soporte PySpark
pip install -r requirements.txt
pip install -r requirements-spark.txt
pip install -e .

# Con herramientas de desarrollo
pip install -e ".[dev]"
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

## Paralelización Automática

TSLib paraleliza automáticamente operaciones computacionalmente intensivas dentro del modelo ARIMA para mejorar el rendimiento con datasets grandes.

### Uso Básico (Paralelización Automática)

```python
# El modelo detecta automáticamente cuándo paralelizar
model = ARIMAModel(order=(1,1,1))
model.fit(data)  # Se paraleliza si la serie es grande
```

### Control Manual

```python
# Deshabilitar paralelización
model = ARIMAModel(order=(1,1,1), n_jobs=1)

# Usar todos los cores disponibles
model = ARIMAModel(order=(1,1,1), n_jobs=-1)

# Usar 4 cores específicamente
model = ARIMAModel(order=(1,1,1), n_jobs=4)
```

### Configuración Óptima

| Tamaño Serie | n_jobs Recomendado | Speedup Esperado |
|--------------|-------------------|------------------|
| < 1000 obs   | 1 (sin paralelo)  | N/A             |
| 1K-10K obs   | 2-4 cores         | 1.5-2x          |
| > 10K obs    | -1 (todos)        | 2-4x            |

### Operaciones Paralelizadas

- **MLE Optimization**: Búsqueda paralela de parámetros iniciales y evaluación de función objetivo
- **ACF/PACF Calculation**: Cálculo paralelo de múltiples lags
- **Gradient Calculation**: Cálculo paralelo de gradientes para optimización
- **Parameter Search**: Evaluación paralela de múltiples conjuntos de parámetros

### Modo Híbrido con Spark

```python
# Spark paraleliza múltiples series
# n_jobs paraleliza operaciones internas
from tslib.spark.parallel_arima import ParallelARIMAProcessor

processor = ParallelARIMAProcessor(n_jobs=2)  # 2 cores por serie
processor.fit_multiple_arima(df, n_jobs=2)
```

### Umbrales de Paralelización

La paralelización se activa automáticamente cuando:
- **MLE Optimization**: > 500 observaciones
- **ACF/PACF**: > 1000 observaciones  
- **Gradient Calculation**: > 1000 observaciones
- **Parameter Search**: > 200 evaluaciones

## Arquitectura del Proyecto

### Estructura Modular

```
tslib/
├── core/              # Algoritmos fundamentales (ARIMA, ACF/PACF, optimización)
│   ├── base.py        # Clases base abstractas (BaseModel, BaseEstimator, etc.)
│   ├── arima.py       # Implementación ARIMA desde cero
│   ├── optimization.py # Motor de optimización MLE
│   └── stationarity.py # Tests de estacionariedad
├── models/            # Interfaces de alto nivel
│   └── arima_model.py # Modelo ARIMA con selección automática
├── preprocessing/     # Transformaciones y validación de datos
├── metrics/           # Métricas de evaluación de modelos
├── spark/             # Integración PySpark
│   ├── parallel_arima.py # Procesamiento paralelo ARIMA
│   └── core.py        # Utilidades Spark
└── utils/             # Funciones utilitarias
```

### Principios de Diseño OOP

La librería sigue una arquitectura orientada a objetos basada en principios SOLID:

- **Single Responsibility**: Cada clase tiene una responsabilidad específica
- **Open/Closed**: Extensible sin modificar código existente
- **Liskov Substitution**: Jerarquía de herencia consistente
- **Interface Segregation**: Interfaces específicas para diferentes funcionalidades
- **Dependency Inversion**: Dependencia de abstracciones, no implementaciones

### Jerarquía de Clases

```
BaseModel (ABC)
├── TimeSeriesModel (ABC)
│   ├── ARProcess
│   ├── MAProcess
│   ├── ARMAProcess
│   └── ARIMAProcess
└── ARIMAModel (High-level interface)

BaseEstimator (ABC)
└── MLEOptimizer

BaseTransformer (ABC)
└── DifferencingTransformer, LogTransformer

SparkEnabled (Mixin)
└── ParallelARIMAProcessor
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

### Comandos Makefile

```bash
# Configuración del entorno de desarrollo
make dev-setup

# Ejecutar tests
make test                    # Todos los tests
make test-coverage          # Con reporte de cobertura
make test-spark             # Tests específicos de Spark
make benchmark              # Benchmarks de rendimiento

# Herramientas de desarrollo
make format                 # Formatear código con black
make lint                   # Linter con flake8
make clean                  # Limpiar archivos temporales

# Ejemplos
make examples               # Ejecutar scripts de ejemplo
```

### Comandos manuales

```bash
# Instalar dependencias de desarrollo
pip install -e ".[dev]"

# Ejecutar tests
pytest tests/ -v

# Tests con cobertura
pytest tests/ --cov=tslib --cov-report=html

# Formatear código
black tslib/ tests/ examples/

# Linter
flake8 tslib/ tests/ examples/
```

## Benchmarks de Rendimiento

La librería incluye tests comprehensivos de rendimiento que comparan la implementación normal vs Spark:

### Métricas Evaluadas

- **Tiempo de ejecución**: Para diferentes tamaños de datasets (1-100 series)
- **Escalabilidad**: Comportamiento con múltiples series temporales
- **Uso de memoria**: Comparación Python vs Spark
- **Precisión**: Verificación de resultados numéricamente idénticos
- **Overhead**: Costo de inicialización de Spark
- **Speedup**: Factor de aceleración y eficiencia

### Ejecutar Benchmarks

```bash
# Benchmarks completos (requiere Java 17+)
make benchmark

# O directamente
python tests/test_performance_benchmark.py
```

**Nota**: Los benchmarks de Spark requieren Java 17+ instalado y configurado correctamente.

### Resultados Típicos

- **Punto de equilibrio**: Spark se vuelve más eficiente con 25+ series
- **Precisión**: Resultados numéricamente idénticos entre implementaciones
- **Escalabilidad**: Spark muestra mejor escalabilidad para datasets grandes

## Documentación

### Documentación Técnica

- [Mathematical Foundations](docs/mathematical_foundations.md) - Fundamentos matemáticos de ARIMA
- [JUSTIFICACION_TECNICA.txt](JUSTIFICACION_TECNICA.txt) - Justificación académica del diseño OOP

### API Reference

#### Modelo ARIMA de Alto Nivel

```python
from tslib.models import ARIMAModel

# Crear modelo con selección automática
model = ARIMAModel(auto_select=True, max_p=5, max_d=2, max_q=5)

# Ajustar modelo
model.fit(data)

# Generar predicciones
forecast = model.predict(steps=10, return_conf_int=True)

# Resumen del modelo
print(model.summary())
```

#### Procesamiento Paralelo con Spark

```python
from tslib.spark import ParallelARIMAProcessor

processor = ParallelARIMAProcessor()

# Procesar múltiples series en paralelo
results = processor.fit_multiple_arima(
    df=spark_df,
    group_column='series_id',
    value_column='value',
    order=(1, 1, 1)
)
```

### Ejemplos

```bash
# Ejecutar ejemplos
make examples

# Ejemplo básico
python examples/basic_arima.py

# Ejemplo con Spark
python examples/spark_parallel_arima.py

# Demo de paralelización interna
python examples/parallel_internal_demo.py
```

## Contribución

### Estructura del Proyecto para Desarrolladores

1. **Core**: Implementa algoritmos matemáticos fundamentales
2. **Models**: Proporciona interfaces de alto nivel
3. **Preprocessing**: Maneja transformaciones de datos
4. **Spark**: Integración con procesamiento distribuido
5. **Tests**: Suite completa de pruebas unitarias e integración

### Guías de Contribución

1. Seguir principios SOLID en el diseño de clases
2. Mantener cobertura de tests >80%
3. Documentar métodos públicos en inglés
4. Usar type hints para mejor mantenibilidad
5. Ejecutar `make lint` antes de commits

### Testing

```bash
# Suite completa de tests
make full-test

# Tests específicos
make test-spark
make benchmark
```

## Licencia

MIT License - Ver [LICENSE](LICENSE) para más detalles.

## Autores

- **Genaro Melgar** - ESCOM, Instituto Politécnico Nacional
  - Implementación ARIMA desde cero
  - Arquitectura orientada a objetos
  - Integración PySpark
  - Tests de rendimiento

## Agradecimientos

- Instituto Politécnico Nacional - ESCOM
- Comunidad de Python para herramientas de desarrollo
- Apache Spark para procesamiento distribuido


