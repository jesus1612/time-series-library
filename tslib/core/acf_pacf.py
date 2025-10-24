"""
Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) calculations

Implements ACF and PACF calculations from scratch using mathematical formulas.
These are essential for identifying ARIMA model orders (p, q).
"""

import numpy as np
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from .base import BaseTest, SparkEnabled
from pyspark.sql.functions import col, lit


class ACFCalculator:
    """
    Calculate Autocorrelation Function (ACF) from scratch
    
    The ACF measures the correlation between a time series and its lagged values.
    Formula: r_k = Σ(y_t - ȳ)(y_{t-k} - ȳ) / Σ(y_t - ȳ)²
    """
    
    def __init__(self, max_lags: Optional[int] = None, n_jobs: int = -1):
        """
        Initialize ACF calculator
        
        Parameters:
        -----------
        max_lags : int, optional
            Maximum number of lags to calculate. If None, uses n/4
        n_jobs : int
            Number of parallel jobs (-1 = all cores, 1 = no parallelization)
        """
        self.max_lags = max_lags
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self._acf_values = None
        self._lags = None
        
        # Umbral para paralelización
        self.parallel_threshold = 1000  # > 1000 observaciones
    
    def calculate(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate ACF for the given time series
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        lags : np.ndarray
            Lag values (0, 1, 2, ...)
        acf_values : np.ndarray
            ACF values for each lag
        """
        data = np.asarray(data)
        n = len(data)
        
        if n < 2:
            raise ValueError("Data must have at least 2 observations")
        
        # Set max_lags if not specified
        if self.max_lags is None:
            max_lags = min(n // 4, 40)  # Standard practice
        else:
            max_lags = min(self.max_lags, n - 1)
        
        # Calculate mean
        mean = np.mean(data)
        
        # Calculate variance (denominator)
        variance = np.sum((data - mean) ** 2)
        
        if variance == 0:
            # Handle constant series
            lags = np.arange(max_lags + 1)
            acf_values = np.ones(max_lags + 1)
            self._lags = lags
            self._acf_values = acf_values
            return lags, acf_values
        
        # Calculate ACF for each lag
        lags = np.arange(max_lags + 1)
        acf_values = np.zeros(max_lags + 1)
        
        # Usar paralelización si el dataset es grande
        if len(data) > self.parallel_threshold and self.n_jobs > 1:
            acf_values = self._calculate_parallel(data, mean, variance, max_lags)
        else:
            acf_values = self._calculate_sequential(data, mean, variance, max_lags)
        
        self._lags = lags
        self._acf_values = acf_values
        return lags, acf_values
    
    def _calculate_sequential(self, data: np.ndarray, mean: float, variance: float, max_lags: int) -> np.ndarray:
        """Calculate ACF sequentially"""
        acf_values = np.zeros(max_lags + 1)
        n = len(data)
        
        for k in range(max_lags + 1):
            if k == 0:
                acf_values[k] = 1.0  # r_0 = 1
            else:
                # Calculate numerator: Σ(y_t - ȳ)(y_{t-k} - ȳ)
                numerator = 0.0
                for t in range(k, n):
                    numerator += (data[t] - mean) * (data[t - k] - mean)
                
                acf_values[k] = numerator / variance
        
        return acf_values
    
    def _calculate_parallel(self, data: np.ndarray, mean: float, variance: float, max_lags: int) -> np.ndarray:
        """Calculate ACF in parallel"""
        def calculate_lag(k):
            """Calculate ACF for a single lag"""
            if k == 0:
                return 1.0
            else:
                # Calculate numerator: Σ(y_t - ȳ)(y_{t-k} - ȳ)
                numerator = np.sum((data[k:] - mean) * (data[:-k] - mean))
                return numerator / variance
        
        # Calcular todos los lags en paralelo
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            acf_values = list(executor.map(calculate_lag, range(max_lags + 1)))
        
        return np.array(acf_values)
    
    def get_acf_values(self) -> Optional[np.ndarray]:
        """Get the calculated ACF values"""
        return self._acf_values
    
    def get_lags(self) -> Optional[np.ndarray]:
        """Get the lag values"""
        return self._lags


class PACFCalculator:
    """
    Calculate Partial Autocorrelation Function (PACF) from scratch
    
    The PACF measures the correlation between y_t and y_{t-k} after removing
    the effects of intermediate lags. Calculated using the Durbin-Levinson algorithm.
    """
    
    def __init__(self, max_lags: Optional[int] = None, n_jobs: int = -1):
        """
        Initialize PACF calculator
        
        Parameters:
        -----------
        max_lags : int, optional
            Maximum number of lags to calculate. If None, uses n/4
        n_jobs : int
            Number of parallel jobs (-1 = all cores, 1 = no parallelization)
        """
        self.max_lags = max_lags
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self._pacf_values = None
        self._lags = None
        
        # Umbral para paralelización
        self.parallel_threshold = 1000  # > 1000 observaciones
    
    def calculate(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate PACF using the Durbin-Levinson algorithm
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        lags : np.ndarray
            Lag values (0, 1, 2, ...)
        pacf_values : np.ndarray
            PACF values for each lag
        """
        data = np.asarray(data)
        n = len(data)
        
        if n < 2:
            raise ValueError("Data must have at least 2 observations")
        
        # Set max_lags if not specified
        if self.max_lags is None:
            max_lags = min(n // 4, 40)  # Standard practice
        else:
            max_lags = min(self.max_lags, n - 1)
        
        # First calculate ACF
        acf_calc = ACFCalculator(max_lags)
        _, acf_values = acf_calc.calculate(data)
        
        # Initialize arrays for Durbin-Levinson algorithm
        lags = np.arange(max_lags + 1)
        pacf_values = np.zeros(max_lags + 1)
        
        # PACF(0) = 1
        pacf_values[0] = 1.0
        
        if max_lags == 0:
            self._lags = lags
            self._pacf_values = pacf_values
            return lags, pacf_values
        
        # PACF(1) = ACF(1)
        pacf_values[1] = acf_values[1]
        
        # Durbin-Levinson algorithm for lags >= 2
        for k in range(2, max_lags + 1):
            # Initialize coefficients
            phi = np.zeros(k)
            phi[k-1] = acf_values[k]  # Initial guess
            
            # Iterative refinement
            for iteration in range(10):  # Max iterations
                phi_old = phi.copy()
                
                # Calculate new coefficients
                for j in range(k-1):
                    phi[j] = phi_old[j] - phi_old[k-1] * phi_old[k-2-j]
                
                # Check convergence
                if np.max(np.abs(phi - phi_old)) < 1e-6:
                    break
            
            pacf_values[k] = phi[k-1]
        
        self._lags = lags
        self._pacf_values = pacf_values
        return lags, pacf_values
    
    def get_pacf_values(self) -> Optional[np.ndarray]:
        """Get the calculated PACF values"""
        return self._pacf_values
    
    def get_lags(self) -> Optional[np.ndarray]:
        """Get the lag values"""
        return self._lags


class ACFPACFAnalyzer:
    """
    Combined ACF/PACF analyzer for model identification
    
    Provides methods to analyze ACF and PACF patterns to suggest
    appropriate ARIMA model orders.
    """
    
    def __init__(self, max_lags: Optional[int] = None):
        """
        Initialize ACF/PACF analyzer
        
        Parameters:
        -----------
        max_lags : int, optional
            Maximum number of lags to calculate
        """
        self.max_lags = max_lags
        self.acf_calc = ACFCalculator(max_lags)
        self.pacf_calc = PACFCalculator(max_lags)
    
    def analyze(self, data: np.ndarray) -> dict:
        """
        Perform complete ACF/PACF analysis
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        results : dict
            Dictionary containing ACF, PACF values and analysis
        """
        # Calculate ACF and PACF
        acf_lags, acf_values = self.acf_calc.calculate(data)
        pacf_lags, pacf_values = self.pacf_calc.calculate(data)
        
        # Suggest model orders based on patterns
        suggested_orders = self._suggest_orders(acf_values, pacf_values)
        
        return {
            'acf_lags': acf_lags,
            'acf_values': acf_values,
            'pacf_lags': pacf_lags,
            'pacf_values': pacf_values,
            'suggested_orders': suggested_orders
        }
    
    def _suggest_orders(self, acf_values: np.ndarray, pacf_values: np.ndarray) -> dict:
        """
        Suggest ARIMA orders based on ACF/PACF patterns
        
        Parameters:
        -----------
        acf_values : np.ndarray
            ACF values
        pacf_values : np.ndarray
            PACF values
            
        Returns:
        --------
        suggestions : dict
            Suggested model orders
        """
        # Calculate significance bounds (approximate 95% confidence)
        n = len(acf_values)
        significance_bound = 1.96 / np.sqrt(n)
        
        # Find significant lags
        significant_acf = np.where(np.abs(acf_values[1:]) > significance_bound)[0] + 1
        significant_pacf = np.where(np.abs(pacf_values[1:]) > significance_bound)[0] + 1
        
        # Suggest AR order (p) - PACF cuts off
        if len(significant_pacf) > 0:
            suggested_p = significant_pacf[0]  # First significant lag
        else:
            suggested_p = 0
        
        # Suggest MA order (q) - ACF cuts off
        if len(significant_acf) > 0:
            suggested_q = significant_acf[0]  # First significant lag
        else:
            suggested_q = 0
        
        return {
            'suggested_p': suggested_p,
            'suggested_q': suggested_q,
            'suggested_d': 0,  # Will be determined by stationarity tests
            'significant_acf_lags': significant_acf.tolist(),
            'significant_pacf_lags': significant_pacf.tolist(),
            'significance_bound': significance_bound
        }


class SparkACFCalculator(SparkEnabled):
    """
    Calculate Autocorrelation Function (ACF) using Spark for distributed computing
    
    Uses Spark to parallelize ACF calculations across multiple lags,
    providing significant performance improvements for large datasets.
    """
    
    def __init__(self, max_lags: Optional[int] = None, spark_session=None, spark_config=None):
        """
        Initialize Spark ACF calculator
        
        Parameters:
        -----------
        max_lags : int, optional
            Maximum number of lags to calculate. If None, uses n/4
        spark_session : SparkSession, optional
            Spark session to use
        spark_config : dict, optional
            Spark configuration
        """
        super().__init__(spark_session, spark_config)
        self.max_lags = max_lags
        self._acf_values = None
        self._lags = None
    
    def calculate(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate ACF for the given time series using Spark
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        lags : np.ndarray
            Lag values (0, 1, 2, ...)
        acf_values : np.ndarray
            ACF values for each lag
        """
        data = np.asarray(data)
        n = len(data)
        
        if n < 2:
            raise ValueError("Data must have at least 2 observations")
        
        # Set max_lags if not specified
        if self.max_lags is None:
            max_lags = min(n // 4, 40)  # Standard practice
        else:
            max_lags = min(self.max_lags, n - 1)
        
        # Convert data to Spark DataFrame
        df_spark = self.converter.to_spark_dataframe(data, cache=True)
        
        # Calculate mean using Spark aggregation
        mean = self.math_ops.vector_mean(df_spark)
        
        # Calculate variance using Spark
        variance = self.math_ops.vector_variance(df_spark, mean=mean)
        
        if variance == 0:
            # Handle constant series
            lags = np.arange(max_lags + 1)
            acf_values = np.ones(max_lags + 1)
            self._lags = lags
            self._acf_values = acf_values
            return lags, acf_values
        
        # Calculate ACF for each lag in parallel
        lags = list(range(max_lags + 1))
        
        # Use Spark to parallelize lag calculations
        lags_rdd = self.spark.sparkContext.parallelize(lags)
        
        def calculate_acf_for_lag(k):
            if k == 0:
                return 1.0
            else:
                # Create lagged data using Spark SQL
                df_lagged = df_spark.alias('orig').join(
                    df_spark.select(
                        (col('index') + lit(k)).alias('index_lag'),
                        col('value').alias('value_lag')
                    ).alias('lag'),
                    col('orig.index') == col('lag.index_lag')
                )
                
                # Calculate covariance
                cov_df = df_lagged.select(
                    ((col('orig.value') - lit(mean)) * 
                     (col('lag.value_lag') - lit(mean))).alias('cov')
                )
                
                covariance = cov_df.agg({'cov': 'mean'}).collect()[0][0]
                return covariance / variance if covariance is not None else 0.0
        
        # Calculate ACF values in parallel
        acf_values = lags_rdd.map(calculate_acf_for_lag).collect()
        
        self._lags = np.array(lags)
        self._acf_values = np.array(acf_values)
        return self._lags, self._acf_values
    
    def get_acf_values(self) -> Optional[np.ndarray]:
        """Get the calculated ACF values"""
        return self._acf_values
    
    def get_lags(self) -> Optional[np.ndarray]:
        """Get the lag values"""
        return self._lags


class SparkPACFCalculator(SparkEnabled):
    """
    Calculate Partial Autocorrelation Function (PACF) using Spark
    
    Uses Spark for distributed computing of PACF calculations,
    particularly beneficial for the Durbin-Levinson algorithm.
    """
    
    def __init__(self, max_lags: Optional[int] = None, spark_session=None, spark_config=None):
        """
        Initialize Spark PACF calculator
        
        Parameters:
        -----------
        max_lags : int, optional
            Maximum number of lags to calculate. If None, uses n/4
        spark_session : SparkSession, optional
            Spark session to use
        spark_config : dict, optional
            Spark configuration
        """
        super().__init__(spark_session, spark_config)
        self.max_lags = max_lags
        self._pacf_values = None
        self._lags = None
    
    def calculate(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate PACF using the Durbin-Levinson algorithm with Spark
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        lags : np.ndarray
            Lag values (0, 1, 2, ...)
        pacf_values : np.ndarray
            PACF values for each lag
        """
        data = np.asarray(data)
        n = len(data)
        
        if n < 2:
            raise ValueError("Data must have at least 2 observations")
        
        # Set max_lags if not specified
        if self.max_lags is None:
            max_lags = min(n // 4, 40)  # Standard practice
        else:
            max_lags = min(self.max_lags, n - 1)
        
        # First calculate ACF using Spark
        acf_calc = SparkACFCalculator(max_lags, self.spark)
        _, acf_values = acf_calc.calculate(data)
        
        # Initialize arrays for Durbin-Levinson algorithm
        lags = np.arange(max_lags + 1)
        pacf_values = np.zeros(max_lags + 1)
        
        # PACF(0) = 1
        pacf_values[0] = 1.0
        
        if max_lags == 0:
            self._lags = lags
            self._pacf_values = pacf_values
            return lags, pacf_values
        
        # PACF(1) = ACF(1)
        pacf_values[1] = acf_values[1]
        
        # Durbin-Levinson algorithm for lags >= 2
        for k in range(2, max_lags + 1):
            # Initialize coefficients
            phi = np.zeros(k)
            phi[k-1] = acf_values[k]  # Initial guess
            
            # Iterative refinement
            for iteration in range(10):  # Max iterations
                phi_old = phi.copy()
                
                # Calculate new coefficients
                for j in range(k-1):
                    phi[j] = phi_old[j] - phi_old[k-1] * phi_old[k-2-j]
                
                # Check convergence
                if np.max(np.abs(phi - phi_old)) < 1e-6:
                    break
            
            pacf_values[k] = phi[k-1]
        
        self._lags = lags
        self._pacf_values = pacf_values
        return lags, pacf_values
    
    def get_pacf_values(self) -> Optional[np.ndarray]:
        """Get the calculated PACF values"""
        return self._pacf_values
    
    def get_lags(self) -> Optional[np.ndarray]:
        """Get the lag values"""
        return self._lags


class SparkACFPACFAnalyzer(SparkEnabled):
    """
    Combined ACF/PACF analyzer using Spark for distributed computing
    
    Provides methods to analyze ACF and PACF patterns using Spark
    to suggest appropriate ARIMA model orders.
    """
    
    def __init__(self, max_lags: Optional[int] = None, spark_session=None, spark_config=None):
        """
        Initialize Spark ACF/PACF analyzer
        
        Parameters:
        -----------
        max_lags : int, optional
            Maximum number of lags to calculate
        spark_session : SparkSession, optional
            Spark session to use
        spark_config : dict, optional
            Spark configuration
        """
        super().__init__(spark_session, spark_config)
        self.max_lags = max_lags
        self.acf_calc = SparkACFCalculator(max_lags, self.spark)
        self.pacf_calc = SparkPACFCalculator(max_lags, self.spark)
    
    def analyze(self, data: np.ndarray) -> dict:
        """
        Perform complete ACF/PACF analysis using Spark
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        results : dict
            Dictionary containing ACF, PACF values and analysis
        """
        # Calculate ACF and PACF using Spark
        acf_lags, acf_values = self.acf_calc.calculate(data)
        pacf_lags, pacf_values = self.pacf_calc.calculate(data)
        
        # Suggest model orders based on patterns
        suggested_orders = self._suggest_orders(acf_values, pacf_values)
        
        return {
            'acf_lags': acf_lags,
            'acf_values': acf_values,
            'pacf_lags': pacf_lags,
            'pacf_values': pacf_values,
            'suggested_orders': suggested_orders
        }
    
    def _suggest_orders(self, acf_values: np.ndarray, pacf_values: np.ndarray) -> dict:
        """
        Suggest ARIMA orders based on ACF/PACF patterns
        
        Parameters:
        -----------
        acf_values : np.ndarray
            ACF values
        pacf_values : np.ndarray
            PACF values
            
        Returns:
        --------
        suggestions : dict
            Suggested model orders
        """
        # Calculate significance bounds (approximate 95% confidence)
        n = len(acf_values)
        significance_bound = 1.96 / np.sqrt(n)
        
        # Find significant lags
        significant_acf = np.where(np.abs(acf_values[1:]) > significance_bound)[0] + 1
        significant_pacf = np.where(np.abs(pacf_values[1:]) > significance_bound)[0] + 1
        
        # Suggest AR order (p) - PACF cuts off
        if len(significant_pacf) > 0:
            suggested_p = significant_pacf[0]  # First significant lag
        else:
            suggested_p = 0
        
        # Suggest MA order (q) - ACF cuts off
        if len(significant_acf) > 0:
            suggested_q = significant_acf[0]  # First significant lag
        else:
            suggested_q = 0
        
        return {
            'suggested_p': suggested_p,
            'suggested_q': suggested_q,
            'suggested_d': 0,  # Will be determined by stationarity tests
            'significant_acf_lags': significant_acf.tolist(),
            'significant_pacf_lags': significant_pacf.tolist(),
            'significance_bound': significance_bound
        }
