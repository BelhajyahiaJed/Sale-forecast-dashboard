import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_forecast_plot(family, historical_data, forecast_exp, forecast_sarima):
    """
    Create a Plotly figure for historical sales and forecasts.
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data.values,
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='blue')
    ))
    
    # Exponential Smoothing forecast
    if not forecast_exp.empty and not forecast_exp.isnull().all():
        fig.add_trace(go.Scatter(
            x=forecast_exp.index,
            y=forecast_exp.values,
            mode='lines',
            name='Exponential Smoothing Forecast',
            line=dict(color='red', dash='dash')
        ))
    
    # SARIMA forecast
    if forecast_sarima is not None and not forecast_sarima.empty and not forecast_sarima.isnull().all():
        fig.add_trace(go.Scatter(
            x=forecast_sarima.index,
            y=forecast_sarima.values,
            mode='lines',
            name='SARIMA Forecast',
            line=dict(color='green', dash='dot')
        ))
    
    fig.update_layout(
        title=f'Sales Forecast for {family}',
        xaxis_title='Date',
        yaxis_title='Quantity',
        hovermode='x unified',
        template='plotly_white',
        xaxis_range=[historical_data.index.min(), pd.Timestamp('2026-05-31')]  # Limit to May 2026
    )
    
    return fig

def create_box_plot(df, family_col, qty_col, title):
    """
    Create a Plotly box plot of quantities by family.
    """
    fig = px.box(
        df,
        x=family_col,
        y=qty_col,
        title=title,
        points='outliers'
    )
    fig.update_layout(
        xaxis_title='Product Family',
        yaxis_title='Ordered Quantity',
        xaxis_tickangle=45,
        template='plotly_white'
    )
    return fig

def create_bar_plot(stats_df, family_col, mean_col, std_col, title):
    """
    Create a Plotly bar plot of mean quantities by family with error bars.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stats_df[family_col],
        y=stats_df[mean_col],
        error_y=dict(
            type='data',
            array=stats_df[std_col],
            visible=True
        ),
        name='Mean Quantity',
        marker_color='skyblue'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Product Family',
        yaxis_title='Mean Quantity',
        xaxis_tickangle=45,
        template='plotly_white'
    )
    return fig

def create_histogram(df, qty_col, title):
    """
    Create a Plotly histogram of quantities.
    """
    fig = px.histogram(
        df,
        x=qty_col,
        nbins=30,
        title=title
    )
    fig.update_layout(
        xaxis_title='Ordered Quantity',
        yaxis_title='Frequency',
        template='plotly_white'
    )
    return fig

def create_time_series_plot(df, date_col, family_col, qty_col, title):
    """
    Create a Plotly time series plot of monthly sales by family.
    """
    df_agg = df.groupby([pd.Grouper(key=date_col, freq='M'), family_col])[qty_col].sum().reset_index()
    fig = px.line(
        df_agg,
        x=date_col,
        y=qty_col,
        color=family_col,
        title=title
    )
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Quantity',
        template='plotly_white'
    )
    return fig

def create_pie_plot(df, family_col, qty_col, title):
    """
    Create a Plotly pie chart of quantity share by family.
    """
    df_agg = df.groupby(family_col)[qty_col].sum().reset_index()
    fig = px.pie(
        df_agg,
        values=qty_col,
        names=family_col,
        title=title
    )
    fig.update_layout(
        template='plotly_white'
    )
    return fig

def create_scatter_plot(stats_df, family_col, mean_col, std_col, title):
    """
    Create a Plotly scatter plot of mean vs. standard deviation by family.
    """
    fig = px.scatter(
        stats_df,
        x=mean_col,
        y=std_col,
        text=family_col,
        title=title
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(
        xaxis_title='Mean Quantity',
        yaxis_title='Standard Deviation',
        template='plotly_white'
    )
    return fig