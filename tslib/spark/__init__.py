"""
PySpark integration module

Provides distributed computing capabilities for time series analysis:
- Parallel ARIMA fitting with Pandas UDF
- Spark utilities and helper functions
"""

from .utils import check_spark_availability

# Only import if PySpark is available
if check_spark_availability():
    from .parallel_arima import fit_predict_arima_udf
    __all__ = ["fit_predict_arima_udf", "check_spark_availability"]
else:
    __all__ = ["check_spark_availability"]

