import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import streamlit as st

# Global flag for debug mode (controlled from dashboard)
DEBUG_MODE = False

def analyze_seasonality_trend(series, period=12):
    """
    Analyze seasonality and trend using seasonal decomposition.
    Returns a dictionary with trend, seasonal, and residual components, and a seasonality flag.
    """
    if len(series) < 2 * period:
        if DEBUG_MODE:
            st.warning(f"Insufficient data for seasonality analysis (less than {2 * period} months). Series length: {len(series)}")
        return {"trend": None, "seasonal": None, "residual": None, "has_seasonality": False}
    
    if DEBUG_MODE:
        st.info(f"Analyzing seasonality and trend for series: min={series.min()}, max={series.max()}, length={len(series)}")
    try:
        decomposition = seasonal_decompose(series, model='additive', period=period)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        seasonal_strength = np.var(seasonal.dropna()) / (np.var(seasonal.dropna()) + np.var(residual.dropna()))
        has_seasonality = seasonal_strength > 0.2
        if DEBUG_MODE:
            st.info(f"Seasonality strength: {seasonal_strength:.3f}, Has seasonality: {has_seasonality}")
        return {"trend": trend, "seasonal": seasonal, "residual": residual, "has_seasonality": has_seasonality}
    except Exception as e:
        if DEBUG_MODE:
            st.warning(f"Seasonality analysis failed: {str(e)}")
        return {"trend": None, "seasonal": None, "residual": None, "has_seasonality": False}

def forecast_dynamic(series, steps=12, period=12):
    """
    Generate dynamic forecast using Exponential Smoothing with trend and seasonality.
    Returns forecast series and model type.
    """
    if len(series) < 2:
        if DEBUG_MODE:
            st.warning(f"Insufficient data for forecasting (less than 2 months). Series length: {len(series)}")
        return pd.Series(), "None"
    
    if DEBUG_MODE:
        st.info(f"Input series for forecasting: min={series.min()}, max={series.max()}, length={len(series)}")
    try:
        model = ExponentialSmoothing(
            series,
            trend='add',
            seasonal='add' if analyze_seasonality_trend(series, period)["has_seasonality"] else None,
            seasonal_periods=period
        )
        fit = model.fit()
        forecast = fit.forecast(steps)
        forecast_index = pd.date_range(start=series.index[-1], periods=steps + 1, freq='M')[1:]
        forecast_series = pd.Series(forecast, index=forecast_index)
        if DEBUG_MODE:
            st.info(f"Forecast generated (Holt-Winters): min={forecast_series.min()}, max={forecast_series.max()}")
        return forecast_series, "holt-winters"
    except Exception as e:
        if DEBUG_MODE:
            st.warning(f"Forecasting failed (Holt-Winters): {str(e)}")
        return pd.Series(), "None"

def forecast_sarima(series, steps=12):
    """
    Generate forecast using SARIMA model.
    Returns forecast series and model type, or None if fitting fails.
    """
    if len(series) < 12:
        if DEBUG_MODE:
            st.warning(f"Insufficient data for SARIMA (less than 12 months). Series length: {len(series)}")
        return None, "None"
    
    if DEBUG_MODE:
        st.info(f"Input series for SARIMA: min={series.min()}, max={series.max()}, length={len(series)}")
    try:
        seasonality = analyze_seasonality_trend(series, 12)["has_seasonality"]
        model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12) if seasonality else (1, 1, 1, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fit = model.fit(disp=False)
        forecast = fit.forecast(steps)
        forecast_index = pd.date_range(start=series.index[-1], periods=steps + 1, freq='M')[1:]
        forecast_series = pd.Series(forecast, index=forecast_index)
        if DEBUG_MODE:
            st.info(f"Forecast generated (SARIMA): min={forecast_series.min()}, max={forecast_series.max()}")
        return forecast_series, "sarima"
    except Exception as e:
        if DEBUG_MODE:
            st.warning(f"Forecasting failed (SARIMA): {str(e)}")
        return None, "None"

def forecast_moving_average(series, steps=12, window=3):
    """
    Generate simple moving average forecast as a fallback.
    """
    if len(series) < window:
        if DEBUG_MODE:
            st.warning(f"Insufficient data for moving average (less than {window} months). Series length: {len(series)}")
        return pd.Series(), "None"
    
    if DEBUG_MODE:
        st.info(f"Input series for moving average: min={series.min()}, max={series.max()}, length={len(series)}")
    try:
        forecast = series.rolling(window=window, min_periods=1).mean().iloc[-1]
        forecast_series = pd.Series([forecast] * steps, index=pd.date_range(start=series.index[-1], periods=steps + 1, freq='M')[1:])
        if DEBUG_MODE:
            st.info(f"Forecast generated (Moving Average): value={forecast}")
        return forecast_series, "moving-average"
    except Exception as e:
        if DEBUG_MODE:
            st.warning(f"Forecasting failed (Moving Average): {str(e)}")
        return pd.Series(), "None"

def forecast_prophet(series, steps=12):
    """
    Generate forecast using Prophet model.
    Returns forecast series and model type.
    """
    if len(series) < 2:
        if DEBUG_MODE:
            st.warning(f"Insufficient data for Prophet (less than 2 months). Series length: {len(series)}")
        return pd.Series(), "None"
    
    if DEBUG_MODE:
        st.info(f"Input series for Prophet: min={series.min()}, max={series.max()}, length={len(series)}")
    # Handle zeros by adding a small constant
    df = pd.DataFrame({'ds': series.index, 'y': series.values})
    df['y'] = df['y'].replace(0, 1e-6)
    
    try:
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df)
        future = model.make_future_dataframe(periods=steps, freq='M')
        forecast = model.predict(future)
        forecast_series = pd.Series(forecast['yhat'].values[-steps:], index=pd.date_range(start=series.index[-1], periods=steps + 1, freq='M')[1:])
        if DEBUG_MODE:
            st.info(f"Forecast generated (Prophet): min={forecast_series.min()}, max={forecast_series.max()}")
        return forecast_series, "prophet"
    except Exception as e:
        if DEBUG_MODE:
            st.warning(f"Forecasting failed (Prophet): {str(e)}")
        return pd.Series(), "None"

def forecast_autoarima(series, steps=12):
    """
    Generate forecast using AutoARIMA model.
    Returns forecast series and model type, or None if fitting fails.
    """
    if len(series) < 12:
        if DEBUG_MODE:
            st.warning(f"Insufficient data for AutoARIMA (less than 12 months). Series length: {len(series)}")
        return None, "None"
    
    if DEBUG_MODE:
        st.info(f"Input series for AutoARIMA: min={series.min()}, max={series.max()}, length={len(series)}")
    # Handle zeros by adding a small constant
    series = series.replace(0, 1e-6)
    
    try:
        model = auto_arima(series, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
        forecast = model.predict(n_periods=steps)
        forecast_index = pd.date_range(start=series.index[-1], periods=steps + 1, freq='M')[1:]
        forecast_series = pd.Series(forecast, index=forecast_index)
        if DEBUG_MODE:
            st.info(f"Forecast generated (AutoARIMA): min={forecast_series.min()}, max={forecast_series.max()}")
        return forecast_series, "autoarima"
    except Exception as e:
        if DEBUG_MODE:
            st.warning(f"Forecasting failed (AutoARIMA): {str(e)}")
        return None, "None"