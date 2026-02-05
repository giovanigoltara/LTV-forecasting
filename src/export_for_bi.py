"""
Export data for Tableau/Looker dashboards.

Creates denormalized, presentation-ready datasets with:
- Pre-calculated metrics and dimensions
- Summary tables for different dashboard views
- Clean column names for easy drag-and-drop
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# Create exports directory
EXPORT_DIR = '/home/user/LTV-forecasting/exports'
os.makedirs(EXPORT_DIR, exist_ok=True)


def load_data():
    """Load and merge all data sources."""
    users = pd.read_csv('/home/user/LTV-forecasting/data/mobile_game_users.csv', parse_dates=['install_date'])
    predictions = pd.read_csv('/home/user/LTV-forecasting/data/ltv_predictions.csv')

    # Merge predictions with user data
    df = users.merge(predictions[['user_id', 'predicted_ltv', 'segment']], on='user_id')
    return df


def create_user_level_export(df):
    """
    Create user-level dataset for detailed analysis.
    Optimized for Tableau/Looker with clean column names.
    """
    export = pd.DataFrame({
        # Identifiers
        'User ID': df['user_id'],
        'Install Date': df['install_date'],
        'Install Month': df['install_date'].dt.to_period('M').astype(str),
        'Install Week': df['install_date'].dt.to_period('W').astype(str),
        'Days Since Install': df['days_since_install'],

        # Dimensions (for filtering/grouping)
        'Acquisition Source': df['acquisition_source'].str.replace('_', ' ').str.title(),
        'Country': df['country'],
        'Platform': df['platform'],
        'Age Group': df['age_group'],

        # Retention Flags
        'Day 1 Retained': df['d1_retention'].map({1: 'Yes', 0: 'No'}),
        'Day 7 Retained': df['d7_retention'].map({1: 'Yes', 0: 'No'}),
        'Day 30 Retained': df['d30_retention'].map({1: 'Yes', 0: 'No'}),

        # Engagement Metrics
        'Total Sessions': df['total_sessions'],
        'Avg Session Duration (min)': df['avg_session_duration_min'].round(1),
        'Total Playtime (hours)': df['total_playtime_hours'].round(1),
        'Levels Completed': df['levels_completed'],
        'Tutorial Completed': df['tutorial_completed'].map({1: 'Yes', 0: 'No'}),
        'Ads Watched': df['ads_watched'],
        'Friends Invited': df['friends_invited'],
        'Guild Joined': df['guild_joined'].map({1: 'Yes', 0: 'No'}),

        # Monetization
        'Is Payer': df['is_payer'].map({1: 'Yes', 0: 'No'}),
        'Number of Purchases': df['num_purchases'],
        'First Purchase Day': df['first_purchase_day'],
        'Total Revenue': df['total_revenue'].round(2),

        # LTV Metrics
        'LTV Day 7': df['ltv_day7'].round(2),
        'LTV Day 30': df['ltv_day30'].round(2),
        'LTV Day 90': df['ltv_day90'].round(2),
        'LTV Day 180': df['ltv_day180'].round(2),
        'LTV Day 365': df['ltv_day365'].round(2),

        # Predictions
        'Predicted LTV': df['predicted_ltv'].round(2),
        'User Segment': df['segment'],

        # Derived Metrics (useful for Tableau calculations)
        'Sessions Per Day': (df['total_sessions'] / np.maximum(df['days_since_install'], 1)).round(2),
        'Revenue Per Session': (df['total_revenue'] / np.maximum(df['total_sessions'], 1)).round(2),
        'Engagement Score': (
            df['d1_retention'] * 10 +
            df['d7_retention'] * 30 +
            df['d30_retention'] * 60 +
            np.log1p(df['total_sessions']) * 5 +
            df['tutorial_completed'] * 10
        ).round(1)
    })

    return export


def create_cohort_summary(df):
    """Create monthly cohort summary for retention/LTV curves."""
    df['cohort'] = df['install_date'].dt.to_period('M').astype(str)

    cohort_summary = df.groupby('cohort').agg({
        'user_id': 'count',
        'd1_retention': 'mean',
        'd7_retention': 'mean',
        'd30_retention': 'mean',
        'is_payer': 'mean',
        'total_revenue': ['sum', 'mean'],
        'predicted_ltv': 'mean',
        'total_sessions': 'mean',
        'total_playtime_hours': 'mean'
    }).round(4)

    cohort_summary.columns = [
        'Total Users',
        'D1 Retention Rate',
        'D7 Retention Rate',
        'D30 Retention Rate',
        'Payer Conversion Rate',
        'Total Revenue',
        'ARPU',
        'Avg Predicted LTV',
        'Avg Sessions',
        'Avg Playtime Hours'
    ]

    cohort_summary = cohort_summary.reset_index()
    cohort_summary.rename(columns={'cohort': 'Cohort Month'}, inplace=True)

    return cohort_summary


def create_acquisition_summary(df):
    """Summary by acquisition source for UA analysis."""
    summary = df.groupby('acquisition_source').agg({
        'user_id': 'count',
        'd1_retention': 'mean',
        'd7_retention': 'mean',
        'is_payer': 'mean',
        'total_revenue': ['sum', 'mean'],
        'predicted_ltv': 'mean',
        'total_sessions': 'mean'
    }).round(4)

    summary.columns = [
        'Total Users',
        'D1 Retention Rate',
        'D7 Retention Rate',
        'Payer Conversion Rate',
        'Total Revenue',
        'ARPU',
        'Avg Predicted LTV',
        'Avg Sessions'
    ]

    summary = summary.reset_index()
    summary.rename(columns={'acquisition_source': 'Acquisition Source'}, inplace=True)
    summary['Acquisition Source'] = summary['Acquisition Source'].str.replace('_', ' ').str.title()

    # Calculate ROI proxy (assuming $2 CPI)
    summary['Est. CPI'] = 2.00
    summary['Est. ROI'] = ((summary['ARPU'] - summary['Est. CPI']) / summary['Est. CPI'] * 100).round(1)

    return summary


def create_country_summary(df):
    """Summary by country for geo targeting."""
    summary = df.groupby('country').agg({
        'user_id': 'count',
        'is_payer': 'mean',
        'total_revenue': ['sum', 'mean'],
        'predicted_ltv': 'mean',
        'd7_retention': 'mean'
    }).round(4)

    summary.columns = [
        'Total Users',
        'Payer Conversion Rate',
        'Total Revenue',
        'ARPU',
        'Avg Predicted LTV',
        'D7 Retention Rate'
    ]

    summary = summary.reset_index()
    summary.rename(columns={'country': 'Country'}, inplace=True)

    # Add country names
    country_names = {
        'US': 'United States', 'JP': 'Japan', 'KR': 'South Korea',
        'DE': 'Germany', 'UK': 'United Kingdom', 'CN': 'China',
        'BR': 'Brazil', 'IN': 'India', 'CA': 'Canada', 'AU': 'Australia'
    }
    summary['Country Name'] = summary['Country'].map(country_names)

    return summary


def create_segment_summary(df):
    """Summary by user segment for whale analysis."""
    summary = df.groupby('segment').agg({
        'user_id': 'count',
        'total_revenue': ['sum', 'mean'],
        'predicted_ltv': 'mean',
        'total_sessions': 'mean',
        'total_playtime_hours': 'mean',
        'num_purchases': 'mean'
    }).round(2)

    summary.columns = [
        'User Count',
        'Total Revenue',
        'Avg Revenue',
        'Avg Predicted LTV',
        'Avg Sessions',
        'Avg Playtime Hours',
        'Avg Purchases'
    ]

    summary = summary.reset_index()
    summary.rename(columns={'segment': 'User Segment'}, inplace=True)

    # Calculate percentages
    total_users = summary['User Count'].sum()
    total_revenue = summary['Total Revenue'].sum()
    summary['% of Users'] = (summary['User Count'] / total_users * 100).round(1)
    summary['% of Revenue'] = (summary['Total Revenue'] / total_revenue * 100).round(1)

    return summary


def create_ltv_curve_data(df):
    """Create data for LTV progression curves."""
    ltv_cols = ['ltv_day7', 'ltv_day30', 'ltv_day90', 'ltv_day180', 'ltv_day365']
    days = [7, 30, 90, 180, 365]

    # Overall LTV curve
    curves = []
    for segment in df['segment'].unique():
        segment_data = df[df['segment'] == segment]
        for day, col in zip(days, ltv_cols):
            curves.append({
                'User Segment': segment,
                'Day': day,
                'Avg LTV': segment_data[col].mean().round(2)
            })

    return pd.DataFrame(curves)


def main():
    print("Loading data...")
    df = load_data()

    print("Creating exports...")

    # 1. User-level detail (main dataset)
    user_export = create_user_level_export(df)
    user_export.to_csv(f'{EXPORT_DIR}/user_level_data.csv', index=False)
    print(f"  ✓ user_level_data.csv ({len(user_export):,} rows)")

    # 2. Cohort summary
    cohort_summary = create_cohort_summary(df)
    cohort_summary.to_csv(f'{EXPORT_DIR}/cohort_summary.csv', index=False)
    print(f"  ✓ cohort_summary.csv ({len(cohort_summary)} rows)")

    # 3. Acquisition source summary
    acq_summary = create_acquisition_summary(df)
    acq_summary.to_csv(f'{EXPORT_DIR}/acquisition_summary.csv', index=False)
    print(f"  ✓ acquisition_summary.csv ({len(acq_summary)} rows)")

    # 4. Country summary
    country_summary = create_country_summary(df)
    country_summary.to_csv(f'{EXPORT_DIR}/country_summary.csv', index=False)
    print(f"  ✓ country_summary.csv ({len(country_summary)} rows)")

    # 5. Segment summary
    segment_summary = create_segment_summary(df)
    segment_summary.to_csv(f'{EXPORT_DIR}/segment_summary.csv', index=False)
    print(f"  ✓ segment_summary.csv ({len(segment_summary)} rows)")

    # 6. LTV curves
    ltv_curves = create_ltv_curve_data(df)
    ltv_curves.to_csv(f'{EXPORT_DIR}/ltv_curves.csv', index=False)
    print(f"  ✓ ltv_curves.csv ({len(ltv_curves)} rows)")

    print(f"\nAll exports saved to: {EXPORT_DIR}/")
    print("\n" + "="*60)
    print("TABLEAU/LOOKER IMPORT GUIDE")
    print("="*60)
    print("""
FILES TO IMPORT:
1. user_level_data.csv    → Main data source (connect first)
2. cohort_summary.csv     → For retention/LTV cohort analysis
3. acquisition_summary.csv → For UA performance dashboard
4. country_summary.csv    → For geo analysis with maps
5. segment_summary.csv    → For whale/segment analysis
6. ltv_curves.csv         → For LTV progression charts

RECOMMENDED JOINS (if using multiple sources):
- Join on 'User Segment' between user_level and segment_summary
- Join on 'Country' between user_level and country_summary
""")


if __name__ == '__main__':
    main()
