# Mathematical Foundations

This document provides the mathematical foundations and theoretical background for the TSLib time series library.

## Table of Contents

1. [ARIMA Models](#arima-models)
2. [Autocorrelation Functions](#autocorrelation-functions)
3. [Stationarity Tests](#stationarity-tests)
4. [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
5. [Information Criteria](#information-criteria)
6. [Forecast Accuracy Metrics](#forecast-accuracy-metrics)

## ARIMA Models

### AutoRegressive (AR) Process

An AR(p) process is defined as:

$$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t$$

Where:
- $y_t$ is the time series value at time $t$
- $c$ is a constant term
- $\phi_1, \phi_2, \ldots, \phi_p$ are the AR parameters
- $\varepsilon_t \sim N(0, \sigma^2)$ is white noise
- $p$ is the order of the AR process

**Stationarity Condition**: The AR process is stationary if all roots of the characteristic polynomial $1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p = 0$ lie outside the unit circle.

### Moving Average (MA) Process

An MA(q) process is defined as:

$$y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q}$$

Where:
- $\mu$ is the mean of the process
- $\theta_1, \theta_2, \ldots, \theta_q$ are the MA parameters
- $\varepsilon_t \sim N(0, \sigma^2)$ is white noise
- $q$ is the order of the MA process

**Invertibility Condition**: The MA process is invertible if all roots of the characteristic polynomial $1 + \theta_1 z + \theta_2 z^2 + \cdots + \theta_q z^q = 0$ lie outside the unit circle.

### ARMA Process

An ARMA(p,q) process combines AR and MA components:

$$y_t = c + \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}$$

### ARIMA Process

An ARIMA(p,d,q) process applies differencing of order $d$ to achieve stationarity:

$$(1 - \phi_1 B - \cdots - \phi_p B^p)(1 - B)^d y_t = c + (1 + \theta_1 B + \cdots + \theta_q B^q)\varepsilon_t$$

Where $B$ is the backshift operator: $B^k y_t = y_{t-k}$.

The differencing operator $(1-B)^d$ is applied $d$ times:
- $d=0$: No differencing
- $d=1$: First difference: $\nabla y_t = y_t - y_{t-1}$
- $d=2$: Second difference: $\nabla^2 y_t = \nabla(\nabla y_t) = y_t - 2y_{t-1} + y_{t-2}$

## Autocorrelation Functions

### Autocorrelation Function (ACF)

The ACF measures the correlation between a time series and its lagged values:

$$\rho_k = \frac{\text{Cov}(y_t, y_{t-k})}{\text{Var}(y_t)} = \frac{\sum_{t=k+1}^n (y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^n (y_t - \bar{y})^2}$$

Where:
- $\rho_k$ is the autocorrelation at lag $k$
- $\bar{y}$ is the sample mean
- $n$ is the sample size

### Partial Autocorrelation Function (PACF)

The PACF measures the correlation between $y_t$ and $y_{t-k}$ after removing the effects of intermediate lags. It is calculated using the Durbin-Levinson algorithm:

1. Initialize: $\phi_{1,1} = \rho_1$
2. For $k = 2, 3, \ldots$:
   $$\phi_{k,k} = \frac{\rho_k - \sum_{j=1}^{k-1} \phi_{k-1,j} \rho_{k-j}}{1 - \sum_{j=1}^{k-1} \phi_{k-1,j} \rho_j}$$
   $$\phi_{k,j} = \phi_{k-1,j} - \phi_{k,k} \phi_{k-1,k-j}, \quad j = 1, 2, \ldots, k-1$$

### Model Identification

- **AR(p)**: PACF cuts off after lag $p$, ACF decays exponentially
- **MA(q)**: ACF cuts off after lag $q$, PACF decays exponentially
- **ARMA(p,q)**: Both ACF and PACF decay exponentially

## Stationarity Tests

### Augmented Dickey-Fuller (ADF) Test

The ADF test examines the null hypothesis that a unit root is present in a time series.

**Test Equation**:
$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + \varepsilon_t$$

Where:
- $\Delta y_t = y_t - y_{t-1}$ is the first difference
- $\alpha$ is a constant term
- $\beta t$ is a time trend
- $\gamma$ is the coefficient on the lagged level
- $\delta_i$ are coefficients on lagged differences

**Hypotheses**:
- $H_0: \gamma = 0$ (unit root present, non-stationary)
- $H_1: \gamma < 0$ (no unit root, stationary)

**Test Statistic**:
$$t_{\text{ADF}} = \frac{\hat{\gamma}}{\text{SE}(\hat{\gamma})}$$

### KPSS Test

The KPSS test examines the null hypothesis that the series is stationary around a deterministic trend.

**Test Statistic**:
$$\text{KPSS} = \frac{\sum_{t=1}^n S_t^2}{n^2 \hat{\sigma}^2}$$

Where:
- $S_t = \sum_{i=1}^t e_i$ is the partial sum of residuals
- $\hat{\sigma}^2$ is the long-run variance estimator
- $e_t$ are the residuals from regressing $y_t$ on a constant and/or trend

**Hypotheses**:
- $H_0$: Series is stationary
- $H_1$: Series has a unit root

## Maximum Likelihood Estimation

### Likelihood Function

For an ARIMA(p,d,q) model, the likelihood function is:

$$L(\boldsymbol{\theta}) = \prod_{t=1}^n f(y_t | \mathcal{F}_{t-1})$$

Where $\boldsymbol{\theta} = (\phi_1, \ldots, \phi_p, \theta_1, \ldots, \theta_q, \sigma^2)$ is the parameter vector.

### Log-Likelihood

$$\ell(\boldsymbol{\theta}) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{t=1}^n \varepsilon_t^2$$

### Optimization

The parameters are estimated by maximizing the log-likelihood:

$$\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta})$$

This is typically done using numerical optimization methods such as:
- BFGS (Broyden-Fletcher-Goldfarb-Shanno)
- L-BFGS-B (Limited-memory BFGS with bounds)
- Newton-Raphson

### Innovation Algorithm

For MA and ARMA models, the innovation algorithm is used to calculate the likelihood:

1. Initialize: $v_0 = \gamma_0$, $\varepsilon_0 = 0$
2. For $t = 1, 2, \ldots, n$:
   $$\varepsilon_t = y_t - \sum_{j=1}^{t-1} \theta_{t-1,j} \varepsilon_{t-j}$$
   $$v_t = v_{t-1} - \frac{\varepsilon_t^2}{v_{t-1}}$$
   $$\theta_{t,j} = \theta_{t-1,j} - \frac{\varepsilon_t \varepsilon_{t-j}}{v_{t-1}}, \quad j = 1, \ldots, t-1$$

## Information Criteria

### Akaike Information Criterion (AIC)

$$\text{AIC} = 2k - 2\ln(L)$$

Where:
- $k$ is the number of parameters
- $L$ is the maximum likelihood value

### Bayesian Information Criterion (BIC)

$$\text{BIC} = k\ln(n) - 2\ln(L)$$

Where $n$ is the number of observations.

### AIC with Correction (AICc)

$$\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n-k-1}$$

### Hannan-Quinn Information Criterion (HQIC)

$$\text{HQIC} = 2k\ln(\ln(n)) - 2\ln(L)$$

**Model Selection**: Choose the model with the lowest information criterion value.

## Forecast Accuracy Metrics

### Root Mean Square Error (RMSE)

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^n (y_t - \hat{y}_t)^2}$$

### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n}\sum_{t=1}^n |y_t - \hat{y}_t|$$

### Mean Absolute Percentage Error (MAPE)

$$\text{MAPE} = \frac{100}{n}\sum_{t=1}^n \left|\frac{y_t - \hat{y}_t}{y_t}\right|$$

### Symmetric Mean Absolute Percentage Error (sMAPE)

$$\text{sMAPE} = \frac{100}{n}\sum_{t=1}^n \frac{2|y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t|}$$

### Mean Absolute Scaled Error (MASE)

$$\text{MASE} = \frac{\text{MAE}}{\text{MAE}_{\text{naive}}}$$

Where $\text{MAE}_{\text{naive}}$ is the MAE of the naive forecast.

### Theil's U Statistic

$$U = \frac{\sqrt{\frac{1}{n}\sum_{t=1}^n (y_t - \hat{y}_t)^2}}{\sqrt{\frac{1}{n}\sum_{t=1}^n y_t^2} + \sqrt{\frac{1}{n}\sum_{t=1}^n \hat{y}_t^2}}$$

## Forecast Variance

For ARIMA models, the forecast variance increases with the forecast horizon:

$$\text{Var}(\hat{y}_{t+h}) = \sigma^2 \sum_{j=0}^{h-1} \psi_j^2$$

Where $\psi_j$ are the coefficients in the MA representation:

$$y_t = \sum_{j=0}^{\infty} \psi_j \varepsilon_{t-j}$$

The $\psi_j$ coefficients are calculated recursively from the AR and MA parameters.

## Confidence Intervals

For a $(1-\alpha) \times 100\%$ confidence interval:

$$\hat{y}_{t+h} \pm z_{\alpha/2} \sqrt{\text{Var}(\hat{y}_{t+h})}$$

Where $z_{\alpha/2}$ is the $(1-\alpha/2)$ quantile of the standard normal distribution.

## References

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. John Wiley & Sons.

2. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.

3. Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*. OTexts.

4. Shumway, R. H., & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications: With R Examples*. Springer.

