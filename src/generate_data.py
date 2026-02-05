"""
Generate synthetic mobile game user data for LTV forecasting.
Based on realistic patterns from mobile gaming industry.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_mobile_game_data(n_users=10000):
    """
    Generate realistic mobile game user data for LTV forecasting.

    Features:
    - User demographics and acquisition info
    - Engagement metrics (sessions, playtime)
    - Monetization (IAP purchases)
    - Retention indicators
    """

    # User acquisition dates (last 365 days)
    base_date = datetime(2025, 1, 1)
    install_dates = [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_users)]

    # Acquisition sources with realistic distribution
    acquisition_sources = np.random.choice(
        ['organic', 'facebook_ads', 'google_ads', 'apple_search', 'influencer', 'cross_promo'],
        n_users,
        p=[0.30, 0.25, 0.20, 0.10, 0.10, 0.05]
    )

    # Countries with realistic mobile gaming distribution
    countries = np.random.choice(
        ['US', 'JP', 'KR', 'DE', 'UK', 'CN', 'BR', 'IN', 'CA', 'AU'],
        n_users,
        p=[0.25, 0.15, 0.10, 0.08, 0.08, 0.12, 0.07, 0.06, 0.05, 0.04]
    )

    # Device platforms
    platforms = np.random.choice(['iOS', 'Android'], n_users, p=[0.45, 0.55])

    # Age groups
    age_groups = np.random.choice(
        ['18-24', '25-34', '35-44', '45-54', '55+'],
        n_users,
        p=[0.25, 0.35, 0.22, 0.12, 0.06]
    )

    # Days since install (for calculating metrics)
    observation_date = datetime(2025, 12, 31)
    days_since_install = [(observation_date - d).days for d in install_dates]

    # Generate engagement metrics with realistic patterns
    # Day 1 retention is typically 25-40%, Day 7 is 10-15%, Day 30 is 3-5%

    d1_retention = np.random.binomial(1, 0.35, n_users)
    d7_retention = np.random.binomial(1, 0.12, n_users) * d1_retention
    d30_retention = np.random.binomial(1, 0.04, n_users) * d7_retention

    # Session counts (power law distribution - most users have few, some have many)
    base_sessions = np.random.exponential(5, n_users) * (1 + d1_retention * 3 + d7_retention * 10 + d30_retention * 30)
    total_sessions = np.clip(base_sessions, 1, 500).astype(int)

    # Average session duration (minutes) - typically 5-15 min for casual games
    avg_session_duration = np.random.gamma(3, 4, n_users) + 2
    avg_session_duration = np.clip(avg_session_duration, 1, 60)

    # Total playtime (hours)
    total_playtime_hours = (total_sessions * avg_session_duration) / 60

    # Levels completed
    levels_completed = np.random.poisson(total_sessions * 0.8, n_users)
    levels_completed = np.clip(levels_completed, 0, 500)

    # Ads watched (for non-payers)
    ads_watched = np.random.poisson(total_sessions * 0.5, n_users)

    # Tutorial completion
    tutorial_completed = np.random.binomial(1, 0.75, n_users)

    # Social features
    friends_invited = np.random.poisson(0.3, n_users) * d7_retention
    guild_joined = np.random.binomial(1, 0.15, n_users) * d7_retention

    # ---- MONETIZATION (Key for LTV) ----
    # Payer conversion rate is typically 2-5% for mobile games

    # Base payer probability influenced by various factors
    payer_prob = np.zeros(n_users)

    # Country effects on spending
    country_multiplier = {
        'US': 1.5, 'JP': 2.0, 'KR': 1.8, 'DE': 1.3, 'UK': 1.4,
        'CN': 1.2, 'BR': 0.6, 'IN': 0.4, 'CA': 1.3, 'AU': 1.4
    }

    # Platform effects (iOS users typically spend more)
    platform_mult = np.where(platforms == 'iOS', 1.3, 1.0)

    # Engagement effects on conversion
    engagement_score = (d1_retention * 0.5 + d7_retention * 1.5 + d30_retention * 3 +
                       tutorial_completed * 0.3 + np.log1p(total_sessions) * 0.1)

    for i in range(n_users):
        base_prob = 0.03  # 3% base conversion
        country_mult = country_multiplier.get(countries[i], 1.0)
        payer_prob[i] = min(0.25, base_prob * country_mult * platform_mult[i] * (1 + engagement_score[i] * 0.1))

    is_payer = np.random.binomial(1, payer_prob)

    # Number of purchases (for payers)
    num_purchases = np.zeros(n_users)
    num_purchases[is_payer == 1] = np.random.negative_binomial(1, 0.3, sum(is_payer)) + 1
    num_purchases = np.clip(num_purchases, 0, 50).astype(int)

    # Revenue per payer (log-normal distribution - typical for mobile games)
    # Whales spend significantly more than average payers
    total_revenue = np.zeros(n_users)

    for i in range(n_users):
        if is_payer[i]:
            # Base spending affected by country and platform
            base_spend = np.random.lognormal(2.5, 1.2) * country_multiplier.get(countries[i], 1.0) * platform_mult[i]
            # Whales (top 1% of payers) spend much more
            if np.random.random() < 0.01:
                base_spend *= np.random.uniform(10, 50)
            total_revenue[i] = base_spend * (1 + engagement_score[i] * 0.2)

    total_revenue = np.round(total_revenue, 2)

    # First purchase day (for payers)
    first_purchase_day = np.zeros(n_users)
    for i in range(n_users):
        if is_payer[i]:
            # Most conversions happen early
            first_purchase_day[i] = min(
                np.random.exponential(7) + 1,  # Avg 7 days to convert
                days_since_install[i]
            )
    first_purchase_day = first_purchase_day.astype(int)

    # Calculate LTV at different time horizons (target variables)
    # This simulates what revenue would be at day 7, 30, 90, 180, 365

    ltv_day7 = total_revenue * np.minimum(7 / np.maximum(days_since_install, 1), 1) * np.random.uniform(0.1, 0.3, n_users)
    ltv_day30 = total_revenue * np.minimum(30 / np.maximum(days_since_install, 1), 1) * np.random.uniform(0.3, 0.5, n_users)
    ltv_day90 = total_revenue * np.minimum(90 / np.maximum(days_since_install, 1), 1) * np.random.uniform(0.5, 0.7, n_users)
    ltv_day180 = total_revenue * np.minimum(180 / np.maximum(days_since_install, 1), 1) * np.random.uniform(0.7, 0.9, n_users)
    ltv_day365 = total_revenue  # Full year LTV

    # Create DataFrame
    df = pd.DataFrame({
        'user_id': [f'user_{i:06d}' for i in range(n_users)],
        'install_date': install_dates,
        'days_since_install': days_since_install,
        'acquisition_source': acquisition_sources,
        'country': countries,
        'platform': platforms,
        'age_group': age_groups,

        # Engagement metrics
        'd1_retention': d1_retention,
        'd7_retention': d7_retention,
        'd30_retention': d30_retention,
        'total_sessions': total_sessions,
        'avg_session_duration_min': np.round(avg_session_duration, 2),
        'total_playtime_hours': np.round(total_playtime_hours, 2),
        'levels_completed': levels_completed,
        'tutorial_completed': tutorial_completed,
        'ads_watched': ads_watched,
        'friends_invited': friends_invited,
        'guild_joined': guild_joined,

        # Monetization
        'is_payer': is_payer,
        'num_purchases': num_purchases,
        'first_purchase_day': first_purchase_day,
        'total_revenue': total_revenue,

        # LTV targets at different horizons
        'ltv_day7': np.round(ltv_day7, 2),
        'ltv_day30': np.round(ltv_day30, 2),
        'ltv_day90': np.round(ltv_day90, 2),
        'ltv_day180': np.round(ltv_day180, 2),
        'ltv_day365': np.round(ltv_day365, 2),
    })

    return df


if __name__ == '__main__':
    print("Generating mobile game user data...")
    df = generate_mobile_game_data(n_users=10000)

    # Save to CSV
    output_path = '/home/user/LTV-forecasting/data/mobile_game_users.csv'
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

    # Print summary statistics
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Total users: {len(df):,}")
    print(f"Payer rate: {df['is_payer'].mean()*100:.2f}%")
    print(f"Average revenue (all users): ${df['total_revenue'].mean():.2f}")
    print(f"Average revenue (payers only): ${df[df['is_payer']==1]['total_revenue'].mean():.2f}")
    print(f"Median LTV (365 days): ${df['ltv_day365'].median():.2f}")
    print(f"Top 10% LTV: ${df['ltv_day365'].quantile(0.9):.2f}")
    print(f"\nRetention rates:")
    print(f"  Day 1: {df['d1_retention'].mean()*100:.1f}%")
    print(f"  Day 7: {df['d7_retention'].mean()*100:.1f}%")
    print(f"  Day 30: {df['d30_retention'].mean()*100:.1f}%")
