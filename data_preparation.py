import pandas as pd
import io
from fuzzywuzzy import process
import streamlit as st

def load_and_preprocess_data(file, file_name="Unknown"):
    """
    Load and preprocess sales data from Excel file or uploaded bytes.
    Returns raw DataFrame and aggregated monthly sales by product family.
    Raises detailed exceptions for debugging.
    """
    try:
        if isinstance(file, str):
            df = pd.read_excel(file)
        else:
            df = pd.read_excel(io.BytesIO(file.read()), engine='openpyxl')
        
        # Log file name for debugging
        st.info(f"Processing file: {file_name}")
        
        # Define required columns based on file name
        if "2021" in file_name.lower() or "2023" in file_name.lower():
            date_col = 'Cmde Date'
            qty_col = 'Qtte Cmdée'
            fam_col = 'Famille'
        else:  # Assume 2024 data
            date_col = 'Créé le'
            qty_col = 'Qtte cmdée'
            fam_col = 'Fam'
        
        # Log selected columns
        #st.info(f"Using columns for {file_name}: Date='{date_col}', Quantity='{qty_col}', Family='{fam_col}'")
        
        required_columns = [date_col, qty_col, fam_col]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            suggestions = {col: process.extractOne(col, df.columns, score_cutoff=80) for col in missing_columns}
            error_msg = f"Missing columns in {file_name}: {missing_columns}. "
            for col, suggestion in suggestions.items():
                if suggestion:
                    error_msg += f"Did you mean '{suggestion[0]}' for '{col}'? "
            raise ValueError(error_msg)

        # Clean and validate
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Log invalid dates
        invalid_dates = df[df[date_col].isna()]
        if not invalid_dates.empty:
            st.warning(f"Invalid dates found in {file_name} (first 5 rows):\n{invalid_dates[[date_col]].head().to_dict()}")
        
        df = df.dropna(subset=[date_col, qty_col])
        df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce')
        
        # Log invalid quantities
        invalid_quantities = df[df[qty_col].isna()]
        if not invalid_quantities.empty:
            st.warning(f"Invalid quantities found in {file_name} (first 5 rows):\n{invalid_quantities[[qty_col]].head().to_dict()}")
        
        df = df.dropna(subset=[qty_col])
        # Cap quantities at 98th percentile
        qty_cap = df[qty_col].quantile(0.98)
        df[qty_col] = df[qty_col].clip(upper=qty_cap)
        df = df[df[qty_col] > 0]  # Ensure positive quantities
        
        # Log capped quantities
        capped_count = (df[qty_col] == qty_cap).sum()
        #if capped_count > 0:
            #st.info(f"Capped {capped_count} quantities at {qty_cap} in {file_name}")
        
        # Rename columns for consistency
        df = df.rename(columns={date_col: 'Date', qty_col: 'Quantity', fam_col: 'Family'})
        
        # Aggregate by month and family, without adding a small constant
        monthly_sales = df.groupby([pd.Grouper(key='Date', freq='M'), 'Family'])['Quantity'].sum().unstack().fillna(0)
        st.info(f"Monthly sales shape for {file_name}: {monthly_sales.shape}")
        #st.info(f"Monthly sales sample for {file_name}:\n{monthly_sales.iloc[:, :5].head()}")  # Show first 5 families
        
        return df, monthly_sales
    
    except Exception as e:
        raise ValueError(f"Error processing {file_name}: {str(e)}")

def get_statistical_description(df, group_col='Family', value_col='Quantity'):
    """
    Compute statistical description for sales data by group.
    Returns DataFrame with mean, std, min, max, etc.
    """
    stats = df.groupby(group_col)[value_col].describe()
    return stats.reset_index()