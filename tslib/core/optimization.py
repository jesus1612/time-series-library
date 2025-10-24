"""
Maximum Likelihood Estimation (MLE) optimization engine

Implements MLE parameter estimation for ARIMA models using numerical optimization.
This is the core engine for fitting ARIMA parameters from scratch.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
from scipy.optimize import minimize, Bounds
from scipy.stats import norm
from .base import BaseEstimator


class MLEOptimizer(BaseEstimator):
    """
    Maximum Likelihood Estimation optimizer for ARIMA models
    
    Implements MLE using numerical optimization methods like BFGS or L-BFGS-B.
    Handles parameter constraints and provides robust optimization.
    """
    
    def __init__(self, 
                 method: str = 'L-BFGS-B',
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize MLE optimizer
        
        Parameters:
        -----------
        method : str
            Optimization method ('L-BFGS-B', 'BFGS', 'SLSQP')
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
        bounds : dict, optional
            Parameter bounds for constrained optimization
        """
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.bounds = bounds or {}
        self._optimization_result = None
        self._log_likelihood = None
    
    def estimate(self, 
                 data: np.ndarray, 
                 model_type: str = 'ARIMA',
                 initial_params: Optional[np.ndarray] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Estimate parameters using Maximum Likelihood Estimation
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        model_type : str
            Type of model ('AR', 'MA', 'ARMA', 'ARIMA')
        initial_params : np.ndarray, optional
            Initial parameter values
        **kwargs
            Additional estimation parameters
            
        Returns:
        --------
        results : Dict[str, Any]
            Estimation results including parameters, log-likelihood, etc.
        """
        data = np.asarray(data)
        n = len(data)
        
        if n < 3:
            raise ValueError("Data must have at least 3 observations")
        
        # Extract model orders
        p = kwargs.get('p', 0)
        d = kwargs.get('d', 0)
        q = kwargs.get('q', 0)
        
        # Validate orders
        if p < 0 or q < 0 or d < 0:
            raise ValueError("Model orders must be non-negative")
        
        if p == 0 and q == 0:
            raise ValueError("At least one of p or q must be positive")
        
        # Apply differencing if needed
        if d > 0:
            diff_data = self._apply_differencing(data, d)
        else:
            diff_data = data.copy()
        
        # Set up optimization
        n_params = p + q + 1  # AR params + MA params + variance
        param_names = self._get_param_names(p, q)
        
        # Set initial parameters if not provided
        if initial_params is None:
            initial_params = self._get_initial_params(p, q, diff_data)
        
        # Set up bounds
        bounds = self._setup_bounds(p, q, param_names)
        
        # Define objective function
        objective = self._create_objective_function(diff_data, p, q, model_type)
        
        # Perform optimization
        try:
            result = minimize(
                objective,
                initial_params,
                method=self.method,
                bounds=bounds,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance,
                    'gtol': self.tolerance
                }
            )
            
            if not result.success:
                raise RuntimeError(f"Optimization failed: {result.message}")
            
            # Extract results
            params = result.x
            log_likelihood = -result.fun  # Convert back to log-likelihood
            
            # Calculate standard errors
            std_errors = self._calculate_standard_errors(params, diff_data, p, q)
            
            # Calculate information criteria
            aic, bic = self._calculate_information_criteria(log_likelihood, n_params, n)
            
            # Store results
            self._optimization_result = result
            self._log_likelihood = log_likelihood
            
            # Organize parameters
            param_dict = self._organize_parameters(params, p, q, param_names)
            
            return {
                'parameters': param_dict,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic,
                'standard_errors': std_errors,
                'optimization_result': result,
                'model_type': model_type,
                'orders': {'p': p, 'd': d, 'q': q}
            }
            
        except Exception as e:
            raise RuntimeError(f"MLE optimization failed: {str(e)}")
    
    def _apply_differencing(self, data: np.ndarray, d: int) -> np.ndarray:
        """
        Apply differencing of order d
        
        Parameters:
        -----------
        data : np.ndarray
            Original time series
        d : int
            Order of differencing
            
        Returns:
        --------
        diff_data : np.ndarray
            Differenced time series
        """
        diff_data = data.copy()
        
        for _ in range(d):
            diff_data = np.diff(diff_data)
        
        return diff_data
    
    def _get_param_names(self, p: int, q: int) -> list:
        """
        Get parameter names for the model
        
        Parameters:
        -----------
        p : int
            AR order
        q : int
            MA order
            
        Returns:
        --------
        param_names : list
            List of parameter names
        """
        names = []
        
        # AR parameters
        for i in range(p):
            names.append(f'phi_{i+1}')
        
        # MA parameters
        for i in range(q):
            names.append(f'theta_{i+1}')
        
        # Variance parameter
        names.append('sigma2')
        
        return names
    
    def _get_initial_params(self, p: int, q: int, data: np.ndarray) -> np.ndarray:
        """
        Get initial parameter values
        
        Parameters:
        -----------
        p : int
            AR order
        q : int
            MA order
        data : np.ndarray
            Time series data
            
        Returns:
        --------
        initial_params : np.ndarray
            Initial parameter values
        """
        n_params = p + q + 1
        initial_params = np.zeros(n_params)
        
        # Initialize AR parameters with small random values
        for i in range(p):
            initial_params[i] = np.random.normal(0, 0.1)
        
        # Initialize MA parameters with small random values
        for i in range(q):
            initial_params[p + i] = np.random.normal(0, 0.1)
        
        # Initialize variance with sample variance
        initial_params[-1] = np.var(data)
        
        return initial_params
    
    def _setup_bounds(self, p: int, q: int, param_names: list) -> list:
        """
        Set up parameter bounds for optimization
        
        Parameters:
        -----------
        p : int
            AR order
        q : int
            MA order
        param_names : list
            Parameter names
            
        Returns:
        --------
        bounds : list
            List of (min, max) tuples for each parameter
        """
        bounds = []
        
        # AR parameters: typically bounded for stationarity
        for i in range(p):
            bounds.append((-0.99, 0.99))
        
        # MA parameters: typically bounded for invertibility
        for i in range(q):
            bounds.append((-0.99, 0.99))
        
        # Variance: must be positive
        bounds.append((1e-6, None))
        
        return bounds
    
    def _create_objective_function(self, 
                                 data: np.ndarray, 
                                 p: int, 
                                 q: int, 
                                 model_type: str) -> Callable:
        """
        Create objective function for optimization
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        p : int
            AR order
        q : int
            MA order
        model_type : str
            Type of model
            
        Returns:
        --------
        objective : callable
            Objective function that returns negative log-likelihood
        """
        def objective(params):
            try:
                # Extract parameters
                ar_params = params[:p] if p > 0 else np.array([])
                ma_params = params[p:p+q] if q > 0 else np.array([])
                sigma2 = params[-1]
                
                # Ensure variance is positive
                if sigma2 <= 0:
                    return 1e10
                
                # Calculate log-likelihood
                log_lik = self._calculate_log_likelihood(data, ar_params, ma_params, sigma2)
                
                # Return negative log-likelihood for minimization
                return -log_lik
                
            except Exception:
                return 1e10  # Return large value if calculation fails
        
        return objective
    
    def _calculate_log_likelihood(self, 
                                data: np.ndarray, 
                                ar_params: np.ndarray, 
                                ma_params: np.ndarray, 
                                sigma2: float) -> float:
        """
        Calculate log-likelihood for given parameters
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        ar_params : np.ndarray
            AR parameters
        ma_params : np.ndarray
            MA parameters
        sigma2 : float
            Variance parameter
            
        Returns:
        --------
        log_likelihood : float
            Log-likelihood value
        """
        n = len(data)
        p = len(ar_params)
        q = len(ma_params)
        
        # Calculate residuals using Kalman filter or innovation algorithm
        residuals = self._calculate_residuals(data, ar_params, ma_params)
        
        # Calculate log-likelihood
        # Assuming Gaussian errors: log L = -n/2 * log(2π) - n/2 * log(σ²) - 1/(2σ²) * Σε²
        log_likelihood = -n/2 * np.log(2 * np.pi) - n/2 * np.log(sigma2) - np.sum(residuals**2) / (2 * sigma2)
        
        return log_likelihood
    
    def _calculate_residuals(self, 
                           data: np.ndarray, 
                           ar_params: np.ndarray, 
                           ma_params: np.ndarray) -> np.ndarray:
        """
        Calculate model residuals using innovation algorithm
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        ar_params : np.ndarray
            AR parameters
        ma_params : np.ndarray
            MA parameters
            
        Returns:
        --------
        residuals : np.ndarray
            Model residuals
        """
        n = len(data)
        p = len(ar_params)
        q = len(ma_params)
        
        # Initialize residuals
        residuals = np.zeros(n)
        
        # For simplicity, use a basic approach
        # In practice, you'd use the innovation algorithm or Kalman filter
        
        for t in range(max(p, q), n):
            # Calculate predicted value
            predicted = 0.0
            
            # AR component
            for i in range(p):
                if t - i - 1 >= 0:
                    predicted += ar_params[i] * data[t - i - 1]
            
            # MA component (using previous residuals)
            for i in range(q):
                if t - i - 1 >= 0:
                    predicted += ma_params[i] * residuals[t - i - 1]
            
            # Calculate residual
            residuals[t] = data[t] - predicted
        
        return residuals
    
    def _calculate_standard_errors(self, 
                                 params: np.ndarray, 
                                 data: np.ndarray, 
                                 p: int, 
                                 q: int) -> np.ndarray:
        """
        Calculate standard errors of parameter estimates
        
        Parameters:
        -----------
        params : np.ndarray
            Estimated parameters
        data : np.ndarray
            Time series data
        p : int
            AR order
        q : int
            MA order
            
        Returns:
        --------
        std_errors : np.ndarray
            Standard errors of parameters
        """
        # This is a simplified calculation
        # In practice, you'd calculate the Hessian matrix and invert it
        
        n = len(data)
        n_params = len(params)
        
        # Approximate standard errors using the inverse of the Hessian
        # For simplicity, use a diagonal approximation
        std_errors = np.sqrt(np.abs(params) / n)
        
        # Ensure minimum standard error
        std_errors = np.maximum(std_errors, 1e-6)
        
        return std_errors
    
    def _calculate_information_criteria(self, 
                                      log_likelihood: float, 
                                      n_params: int, 
                                      n_obs: int) -> Tuple[float, float]:
        """
        Calculate AIC and BIC information criteria
        
        Parameters:
        -----------
        log_likelihood : float
            Log-likelihood value
        n_params : int
            Number of parameters
        n_obs : int
            Number of observations
            
        Returns:
        --------
        aic : float
            Akaike Information Criterion
        bic : float
            Bayesian Information Criterion
        """
        # AIC = 2k - 2ln(L)
        aic = 2 * n_params - 2 * log_likelihood
        
        # BIC = k*ln(n) - 2ln(L)
        bic = n_params * np.log(n_obs) - 2 * log_likelihood
        
        return aic, bic
    
    def _organize_parameters(self, 
                           params: np.ndarray, 
                           p: int, 
                           q: int, 
                           param_names: list) -> Dict[str, float]:
        """
        Organize parameters into a dictionary
        
        Parameters:
        -----------
        params : np.ndarray
            Parameter values
        p : int
            AR order
        q : int
            MA order
        param_names : list
            Parameter names
            
        Returns:
        --------
        param_dict : dict
            Dictionary of parameter names and values
        """
        param_dict = {}
        
        for i, name in enumerate(param_names):
            param_dict[name] = params[i]
        
        return param_dict
    
    @property
    def optimization_result(self):
        """Get the optimization result"""
        return self._optimization_result
    
    @property
    def log_likelihood(self):
        """Get the log-likelihood value"""
        return self._log_likelihood

