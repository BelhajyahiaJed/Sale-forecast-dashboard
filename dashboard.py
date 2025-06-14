import streamlit as st
import pandas as pd
from data_preparation import load_and_preprocess_data, get_statistical_description
from abc_xyz_classification import classement_abc, classement_xyz
from forecasting import forecast_dynamic, forecast_sarima, forecast_moving_average, forecast_prophet, forecast_autoarima, analyze_seasonality_trend
from evaluation import calculate_error_metrics
from visualization import create_forecast_plot, create_box_plot, create_bar_plot, create_histogram, create_time_series_plot, create_pie_plot, create_scatter_plot
import numpy as np

def main():
    # Session state for page navigation, debug mode, and data
    if 'page' not in st.session_state:
        st.session_state.page = "Welcome"
    if 'DEBUG_MODE' not in st.session_state:
        st.session_state.DEBUG_MODE = False
    if 'df_hist' not in st.session_state:
        st.session_state.df_hist = None
    if 'monthly_sales_hist' not in st.session_state:
        st.session_state.monthly_sales_hist = None
    if 'df_2024' not in st.session_state:
        st.session_state.df_2024 = None
    if 'monthly_sales_2024' not in st.session_state:
        st.session_state.monthly_sales_2024 = None
    if 'historical_file' not in st.session_state:
        st.session_state.historical_file = None
    if 'actual_file' not in st.session_state:
        st.session_state.actual_file = None

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Welcome", "Data Overview", "Statistical Description and Visualizations", "Classifications", "Forecasting", "Comparison"])

    # Update session state based on sidebar selection
    st.session_state.page = page

    # File uploads and debug mode on Welcome page
    if st.session_state.page == "Welcome":
        st.header("Welcome to the Sales Forecasting Dashboard")
        st.markdown("""
            This dashboard is designed for Boukhari Prince Medical Industry to analyze sales data, 
            generate forecasts, and validate them against actual 2024 sales. It provides:
            - **Statistical Descriptions**: Insights into 2021–2023 and 2024 data.
            - **ABC/XYZ Classifications**: Categorization of product families.
            - **Forecasting**: Sales predictions with error metrics for 2024.
            - **Data Overview**: Summary of loaded data.
            - **Visualizations**: Customizable plots for data exploration.
            - **Comparison**: Forecast vs. actual data analysis.

            **Instructions**:
            1. Upload your 2021–2023 sales data (training set) and 2024 sales data (validation set) below.
            2. Use the sidebar to navigate to different sections.
            3. Select a product family to view detailed forecasts and metrics.

            **Note**: Ensure your Excel files contain columns for 'Date', 'Quantity', and 'Family' (or their equivalents).
        """)
        st.session_state.historical_file = st.file_uploader("Upload 2021–2023 Sales Data (Excel)", type=['xlsx'], key="hist_file")
        st.session_state.actual_file = st.file_uploader("Upload 2024 Sales Data (Excel)", type=['xlsx'], key="2024_file")
        DEBUG_MODE = st.checkbox("Enable Debug Mode", value=st.session_state.DEBUG_MODE, key="debug_mode_welcome")
        st.session_state.DEBUG_MODE = DEBUG_MODE

        # Load data if files are uploaded
        if st.session_state.historical_file and st.session_state.df_hist is None:
            try:
                st.session_state.df_hist, st.session_state.monthly_sales_hist = load_and_preprocess_data(st.session_state.historical_file, file_name="2021–2023 Data")
            except ValueError as e:
                st.error(str(e))
        if st.session_state.actual_file and st.session_state.df_2024 is None:
            try:
                st.session_state.df_2024, st.session_state.monthly_sales_2024 = load_and_preprocess_data(st.session_state.actual_file, file_name="2024 Data")
            except ValueError as e:
                st.error(str(e))

    # Debug Mode from session state
    DEBUG_MODE = st.session_state.DEBUG_MODE

    # Access loaded data
    df_hist = st.session_state.df_hist
    monthly_sales_hist = st.session_state.monthly_sales_hist
    df_2024 = st.session_state.df_2024
    monthly_sales_2024 = st.session_state.monthly_sales_2024

    # Page content
    st.title(st.session_state.page)

    if st.session_state.page == "Welcome":
        if not st.session_state.historical_file or not st.session_state.actual_file:
            st.warning("Please upload both 2021–2023 and 2024 sales data files to proceed with analysis.")

    elif st.session_state.page == "Data Overview":
        st.header("Data Overview")
        if df_hist is None or df_hist.empty:
            st.warning("Please upload the 2021–2023 sales data file to proceed.")
        else:
            st.subheader("2021–2023 Data")
            st.write(f"Data shape: {df_hist.shape}")
            st.write("Sample data:")
            st.dataframe(df_hist.head())
            if monthly_sales_hist is not None:
                st.write(f"Monthly sales shape: {monthly_sales_hist.shape}")
                st.write("Monthly sales sample:")
                st.dataframe(monthly_sales_hist.head())
        if df_2024 is None or df_2024.empty:
            st.warning("Please upload the 2024 sales data file to proceed.")
        else:
            st.subheader("2024 Data")
            st.write(f"Data shape: {df_2024.shape}")
            st.write("Sample data:")
            st.dataframe(df_2024.head())
            if monthly_sales_2024 is not None:
                st.write(f"Monthly sales shape: {monthly_sales_2024.shape}")
                st.write("Monthly sales sample:")
                st.dataframe(monthly_sales_2024.head())

    elif st.session_state.page == "Statistical Description and Visualizations":
        if df_hist is None or df_hist.empty:
            st.warning("Please upload the 2021–2023 sales data file to proceed.")
        else:
            st.header("Statistical Description and Visualizations")
            
            # 2021–2023 Data
            st.subheader("2021–2023 Data")
            stats_hist = get_statistical_description(df_hist)
            st.dataframe(stats_hist.style.format(precision=2))
            
            # Optional plot selection
            st.subheader("Select Visualizations for 2021–2023")
            plot_options = {
                "Box Plot of Quantities by Family": True,
                "Bar Plot of Mean Quantities by Family": True,
                "Histogram of Quantities": True,
                "Time Series Plot of Monthly Sales": False,
                "Pie Chart of Quantity Share by Family": False,
                "Scatter Plot of Mean vs. Std Dev": False
            }
            for plot_name, default in plot_options.items():
                plot_options[plot_name] = st.checkbox(plot_name, value=default)
            
            # Display selected plots
            if any(plot_options.values()):
                col1, col2 = st.columns(2)
                with col1:
                    if plot_options["Box Plot of Quantities by Family"]:
                        st.subheader("Box Plot of Quantities by Family")
                        fig_box_hist = create_box_plot(df_hist, 'Family', 'Quantity', '2021–2023 Quantities by Family')
                        st.plotly_chart(fig_box_hist, use_container_width=True)
                    if plot_options["Time Series Plot of Monthly Sales"]:
                        st.subheader("Time Series Plot of Monthly Sales")
                        fig_ts_hist = create_time_series_plot(df_hist, 'Date', 'Family', 'Quantity', '2021–2023 Monthly Sales')
                        st.plotly_chart(fig_ts_hist, use_container_width=True)
                
                with col2:
                    if plot_options["Bar Plot of Mean Quantities by Family"]:
                        st.subheader("Bar Plot of Mean Quantities by Family")
                        fig_bar_hist = create_bar_plot(stats_hist, 'Family', 'mean', 'std', '2021–2023 Mean Quantities')
                        st.plotly_chart(fig_bar_hist, use_container_width=True)
                    if plot_options["Pie Chart of Quantity Share by Family"]:
                        st.subheader("Pie Chart of Quantity Share by Family")
                        fig_pie_hist = create_pie_plot(df_hist, 'Family', 'Quantity', '2021–2023 Quantity Share')
                        st.plotly_chart(fig_pie_hist, use_container_width=True)
                
                if plot_options["Histogram of Quantities"]:
                    st.subheader("Histogram of Quantities")
                    fig_hist_hist = create_histogram(df_hist, 'Quantity', '2021–2023 Quantity Distribution')
                    st.plotly_chart(fig_hist_hist, use_container_width=True)
                if plot_options["Scatter Plot of Mean vs. Std Dev"]:
                    st.subheader("Scatter Plot of Mean vs. Std Dev")
                    fig_scatter_hist = create_scatter_plot(stats_hist, 'Family', 'mean', 'std', '2021–2023 Mean vs. Std Dev')
                    st.plotly_chart(fig_scatter_hist, use_container_width=True)

            if df_2024 is not None and not df_2024.empty:
                st.subheader("2024 Data")
                stats_2024 = get_statistical_description(df_2024)
                st.dataframe(stats_2024.style.format(precision=2))
                
                # Optional plot selection for 2024
                st.subheader("Select Visualizations for 2024")
                plot_options_2024 = {f"{k} (2024)": v for k, v in plot_options.items()}
                for plot_name, default in plot_options_2024.items():
                    plot_options_2024[plot_name] = st.checkbox(plot_name, value=default)
                
                # Display selected 2024 plots
                if any(plot_options_2024.values()):
                    col1, col2 = st.columns(2)
                    with col1:
                        if plot_options_2024["Box Plot of Quantities by Family (2024)"]:
                            st.subheader("Box Plot of Quantities by Family (2024)")
                            fig_box_2024 = create_box_plot(df_2024, 'Family', 'Quantity', '2024 Quantities by Family')
                            st.plotly_chart(fig_box_2024, use_container_width=True)
                        if plot_options_2024["Time Series Plot of Monthly Sales (2024)"]:
                            st.subheader("Time Series Plot of Monthly Sales (2024)")
                            fig_ts_2024 = create_time_series_plot(df_2024, 'Date', 'Family', 'Quantity', '2024 Monthly Sales')
                            st.plotly_chart(fig_ts_2024, use_container_width=True)
                    
                    with col2:
                        if plot_options_2024["Bar Plot of Mean Quantities by Family (2024)"]:
                            st.subheader("Bar Plot of Mean Quantities by Family (2024)")
                            fig_bar_2024 = create_bar_plot(stats_2024, 'Family', 'mean', 'std', '2024 Mean Quantities')
                            st.plotly_chart(fig_bar_2024, use_container_width=True)
                        if plot_options_2024["Pie Chart of Quantity Share by Family (2024)"]:
                            st.subheader("Pie Chart of Quantity Share by Family (2024)")
                            fig_pie_2024 = create_pie_plot(df_2024, 'Family', 'Quantity', '2024 Quantity Share')
                            st.plotly_chart(fig_pie_2024, use_container_width=True)
                    
                    if plot_options_2024["Histogram of Quantities (2024)"]:
                        st.subheader("Histogram of Quantities (2024)")
                        fig_hist_2024 = create_histogram(df_2024, 'Quantity', '2024 Quantity Distribution')
                        st.plotly_chart(fig_hist_2024, use_container_width=True)
                    if plot_options_2024["Scatter Plot of Mean vs. Std Dev (2024)"]:
                        st.subheader("Scatter Plot of Mean vs. Std Dev (2024)")
                        fig_scatter_2024 = create_scatter_plot(stats_2024, 'Family', 'mean', 'std', '2024 Mean vs. Std Dev')
                        st.plotly_chart(fig_scatter_2024, use_container_width=True)

    elif st.session_state.page == "Classifications":
        if df_hist is None or df_hist.empty:
            st.warning("Please upload the 2021–2023 sales data file to proceed.")
        else:
            st.header("ABC and XYZ Classifications")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ABC Classification")
                abc = classement_abc(df_hist, article_col='Family', qty_col='Quantity')
                st.dataframe(abc)
            with col2:
                st.subheader("XYZ Classification")
                xyz = classement_xyz(df_hist, article_col='Family', qty_col='Quantity')
                st.dataframe(xyz)
            if DEBUG_MODE:
                st.subheader("Debug: Classification Summary")
                st.write(f"Number of Class A families: {len(abc[abc['Classe_ABC'] == 'A'])}")
                st.write(f"Number of Class X/Y families: {len(xyz[xyz['Classe_XYZ'].isin(['X', 'Y'])])}")

    elif st.session_state.page == "Forecasting":
        if df_hist is None or df_hist.empty or df_2024 is None or df_2024.empty:
            st.warning("Please upload both 2021–2023 and 2024 sales data files to proceed with forecasting.")
        else:
            st.header("Sales Forecasts")
            
            # Target families
            abc = classement_abc(df_hist, article_col='Family', qty_col='Quantity')
            xyz = classement_xyz(df_hist, article_col='Family', qty_col='Quantity')
            target_families = abc[abc['Classe_ABC'] == 'A'].merge(xyz[xyz['Classe_XYZ'].isin(['X', 'Y'])], on='Family')['Family'].tolist()
            if not target_families:
                st.warning("No families are both Class A and X/Y. Including all product families for forecasting.")
                target_families = sorted(df_hist['Family'].dropna().unique().tolist())
            
            if DEBUG_MODE:
                st.write(f"Debug: Available families for forecasting: {target_families}")

            if not target_families:
                st.error("No valid families available for forecasting. Check data for missing or invalid 'Famille' values.")
                return  # Early exit, no return value needed

            family = st.selectbox("Select Product Family", target_families)

            if family:
                series_hist = monthly_sales_hist.get(family, pd.Series())
                if DEBUG_MODE:
                    st.write(f"Debug: Historical data length for {family}: {len(series_hist)} months")
                if len(series_hist) < 24:
                    st.warning(f"Insufficient historical data for {family} (requires at least 24 months). Try aggregating data or reducing the threshold.")
                    return  # Early exit, no return value needed

                # Analyze seasonality and trend
                analysis = analyze_seasonality_trend(series_hist)
                st.subheader(f"Seasonality and Trend Analysis for {family}")
                if analysis["trend"] is not None:
                    st.write(f"Has Seasonality: {analysis['has_seasonality']}")
                    st.write("Trend Component (sample):", analysis["trend"].dropna().head())
                    st.write("Seasonal Component (sample):", analysis["seasonal"].dropna().head())

                # Use full historical data for training
                train = series_hist
                test = None
                if monthly_sales_2024 is not None and not monthly_sales_2024.empty:
                    if DEBUG_MODE:
                        st.write(f"Debug: 2024 families available: {monthly_sales_2024.columns.tolist()}")
                    # Standardize family names for matching
                    family_std = family.strip().lower()
                    available_2024_families = [f.strip().lower() for f in monthly_sales_2024.columns]
                    if family_std in available_2024_families:
                        matching_family = monthly_sales_2024.columns[available_2024_families.index(family_std)]
                        test = monthly_sales_2024[matching_family]
                        if DEBUG_MODE:
                            st.write(f"Debug: 2024 data for {family} loaded, length: {len(test)}")
                        if len(test) < 6:
                            st.warning(f"2024 data for {family} has only {len(test)} months, insufficient for detailed metrics.")
                    else:
                        if DEBUG_MODE:
                            st.warning(f"Debug: No 2024 data found for family {family}, available: {available_2024_families}")

                # Try multiple forecasting methods and store results
                forecasts = {
                    "holt-winters": forecast_dynamic(train, steps=6),
                    "sarima": forecast_sarima(train, steps=6),
                    "prophet": forecast_prophet(train, steps=6),
                    "autoarima": forecast_autoarima(train, steps=6),
                    "moving-average": forecast_moving_average(train, steps=6, window=3)
                }

                # Select the first valid forecast and enforce non-negative values
                forecast_exp = pd.Series()
                model_type = "None"
                for model, (forecast, m_type) in forecasts.items():
                    if not forecast.empty and not forecast.isnull().all():
                        forecast_exp = forecast.clip(lower=0)
                        model_type = m_type
                        break

                # Display forecast plot
                st.subheader(f"{family} Forecast (Model: {model_type})")
                if forecast_exp.empty or forecast_exp.isnull().all():
                    st.warning("No valid forecast generated. Check data or model warnings above.")
                else:
                    sarima_forecast = forecasts["sarima"][0].clip(lower=0) if not forecasts["sarima"][0].empty and not forecasts["sarima"][0].isnull().all() else pd.Series()
                    fig = create_forecast_plot(family, monthly_sales_hist[family], forecast_exp, sarima_forecast)
                    st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.page == "Comparison":
        if df_hist is None or df_hist.empty or df_2024 is None or df_2024.empty:
            st.warning("Please upload both 2021–2023 and 2024 sales data files to proceed with comparison.")
        else:
            st.header("Forecast vs. Actual Comparison")
            
            # Target families
            abc = classement_abc(df_hist, article_col='Family', qty_col='Quantity')
            xyz = classement_xyz(df_hist, article_col='Family', qty_col='Quantity')
            target_families = abc[abc['Classe_ABC'] == 'A'].merge(xyz[xyz['Classe_XYZ'].isin(['X', 'Y'])], on='Family')['Family'].tolist()
            if not target_families:
                st.warning("No families are both Class A and X/Y. Including all product families for comparison.")
                target_families = sorted(df_hist['Family'].dropna().unique().tolist())
            
            if DEBUG_MODE:
                st.write(f"Debug: Available families for comparison: {target_families}")

            if not target_families:
                st.error("No valid families available for comparison. Check data for missing or invalid 'Famille' values.")
                return  # Early exit, no return value needed

            family = st.selectbox("Select Product Family", target_families)

            if family:
                series_hist = monthly_sales_hist.get(family, pd.Series())
                if DEBUG_MODE:
                    st.write(f"Debug: Historical data length for {family}: {len(series_hist)} months")
                if len(series_hist) < 24:
                    st.warning(f"Insufficient historical data for {family} (requires at least 24 months). Try aggregating data or reducing the threshold.")
                    return  # Early exit, no return value needed

                # Use full historical data for training
                train = series_hist
                test = None
                if monthly_sales_2024 is not None and not monthly_sales_2024.empty:
                    if DEBUG_MODE:
                        st.write(f"Debug: 2024 families available: {monthly_sales_2024.columns.tolist()}")
                    # Standardize family names for matching
                    family_std = family.strip().lower()
                    available_2024_families = [f.strip().lower() for f in monthly_sales_2024.columns]
                    if family_std in available_2024_families:
                        matching_family = monthly_sales_2024.columns[available_2024_families.index(family_std)]
                        test = monthly_sales_2024[matching_family]
                        if DEBUG_MODE:
                            st.write(f"Debug: 2024 data for {family} loaded, length: {len(test)}")
                    else:
                        if DEBUG_MODE:
                            st.warning(f"Debug: No 2024 data found for family {family}, available: {available_2024_families}")

                if test is not None and len(test) > 0:
                    # Try multiple forecasting methods and store results
                    forecasts = {
                        "holt-winters": forecast_dynamic(train, steps=len(test)),
                        "sarima": forecast_sarima(train, steps=len(test)),
                        "prophet": forecast_prophet(train, steps=len(test)),
                        "autoarima": forecast_autoarima(train, steps=len(test)),
                        "moving-average": forecast_moving_average(train, steps=len(test), window=3)
                    }

                    # Compare forecasts with actual data
                    st.subheader(f"Comparison for {family}")
                    for model, (forecast, model_type) in forecasts.items():
                        if not forecast.empty and not forecast.isnull().all():
                            forecast = forecast[:len(test)].clip(lower=0)
                            actual = test.values
                            if len(test) >= 1 and not np.all(actual == 0):
                                errors = calculate_error_metrics(actual, forecast.values)
                                st.write(f"**{model_type} Forecast vs. Actual**")
                                fig = create_forecast_plot(family, pd.concat([series_hist, test]), forecast, pd.Series(), title=f"{model_type} vs. Actual")
                                st.plotly_chart(fig, use_container_width=True)
                                st.write(f"MAE: {errors['MAE']:.2f}, RMSE: {errors['RMSE']:.2f}, MAPE: {errors['MAPE']:.2f}%")
                            else:
                                st.info(f"{model_type} data insufficient or all zero for error metrics.")
                else:
                    st.info("No 2024 data available for comparison.")

if __name__ == "__main__":
    main()