import pandas as pd
import numpy as np

def classement_abc(df, article_col='Famille', qty_col='Qtte Cmdée'):
    """
    Perform ABC classification based on cumulative sales contribution.
    Returns DataFrame with article IDs and ABC classes.
    """
    total_per_article = df.groupby(article_col)[qty_col].sum().sort_values(ascending=False)
    total_cum = total_per_article.cumsum()
    total_sum = total_per_article.sum()
    pct_cum = total_cum / total_sum
    abc = pct_cum.apply(lambda x: 'A' if x <= 0.7 else 'B' if x <= 0.9 else 'C')
    return abc.reset_index().rename(columns={qty_col: 'Classe_ABC'})

def classement_xyz(df, article_col='Famille', qty_col='Qtte Cmdée'):
    """
    Perform XYZ classification based on coefficient of variation.
    Returns DataFrame with article IDs and XYZ classes.
    """
    stats = df.groupby(article_col)[qty_col].agg(['mean', 'std'])
    stats['cv'] = stats['std'] / stats['mean']
    stats = stats.fillna(0)
    stats['Classe_XYZ'] = stats['cv'].apply(lambda cv: 'X' if cv < 0.1 else 'Y' if cv < 0.25 else 'Z')
    return stats[['Classe_XYZ']].reset_index()