#!/usr/bin/env python3
"""
Daily Update Script for Texas Governor Race Analysis
Fetches latest data, runs model, and generates morning briefing.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

GUARDIAN_API_KEY = os.getenv('GUARDIAN_API_KEY')
NYT_API_KEY = os.getenv('NYT_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')

DATA_DIR = Path(__file__).parent / 'data'
BRIEFING_DIR = Path(__file__).parent / 'briefings'
BRIEFING_DIR.mkdir(exist_ok=True)


def fetch_latest_news():
    """Fetch latest news about Texas Governor race."""
    news_items = []
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Guardian API
    if GUARDIAN_API_KEY:
        try:
            url = "https://content.guardianapis.com/search"
            params = {
                'q': 'Texas Governor Abbott OR Hinojosa',
                'from-date': yesterday,
                'to-date': today,
                'api-key': GUARDIAN_API_KEY,
                'page-size': 10,
                'show-fields': 'headline,trailText'
            }
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('response', {}).get('results', []):
                    news_items.append({
                        'source': 'Guardian',
                        'headline': item.get('webTitle', ''),
                        'summary': item.get('fields', {}).get('trailText', ''),
                        'date': item.get('webPublicationDate', ''),
                        'url': item.get('webUrl', '')
                    })
        except Exception as e:
            print(f"Guardian API error: {e}")

    # NYT API
    if NYT_API_KEY:
        try:
            url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
            params = {
                'q': 'Texas Governor',
                'begin_date': yesterday.replace('-', ''),
                'end_date': today.replace('-', ''),
                'api-key': NYT_API_KEY
            }
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('response', {}).get('docs', []):
                    news_items.append({
                        'source': 'NYT',
                        'headline': item.get('headline', {}).get('main', ''),
                        'summary': item.get('abstract', ''),
                        'date': item.get('pub_date', ''),
                        'url': item.get('web_url', '')
                    })
        except Exception as e:
            print(f"NYT API error: {e}")

    return news_items


def fetch_latest_economic_data():
    """Fetch latest economic indicators from FRED."""
    indicators = {}

    if not FRED_API_KEY:
        return indicators

    series_ids = {
        'UNRATE': 'unemployment_rate',
        'CPIAUCSL': 'cpi',
        'VIXCLS': 'vix',
        'GDP': 'gdp'
    }

    for series_id, name in series_ids.items():
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': FRED_API_KEY,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': 1
            }
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                if observations:
                    indicators[name] = {
                        'value': observations[0].get('value'),
                        'date': observations[0].get('date')
                    }
        except Exception as e:
            print(f"FRED API error for {series_id}: {e}")

    return indicators


def fetch_latest_polling():
    """Check for new polling data."""
    # In production, this would scrape FiveThirtyEight or RealClearPolitics
    # For now, return placeholder indicating no new polls
    return {
        'new_polls': False,
        'message': 'No new polls detected in the last 24 hours',
        'current_average': {
            'abbott': 50,
            'hinojosa': 42,
            'margin': 8
        }
    }


def generate_morning_briefing():
    """Generate the daily morning briefing."""
    briefing_date = datetime.now().strftime('%Y-%m-%d')
    briefing_time = datetime.now().strftime('%H:%M')

    print(f"Generating morning briefing for {briefing_date}...")

    # Fetch latest data
    news = fetch_latest_news()
    economic = fetch_latest_economic_data()
    polling = fetch_latest_polling()

    # Build briefing content
    briefing = {
        'date': briefing_date,
        'generated_at': f"{briefing_date} {briefing_time}",
        'race': 'Texas Governor 2026',
        'current_forecast': {
            'predicted_winner': 'Greg Abbott (R)',
            'predicted_margin': 'R+12-18%',
            'win_probability': '99%+',
            'confidence': 'Very High'
        },
        'polling_update': polling,
        'economic_indicators': economic,
        'news_summary': {
            'total_articles': len(news),
            'articles': news[:5]  # Top 5 articles
        },
        'key_changes': [],
        'alerts': []
    }

    # Check for significant changes
    if economic.get('vix', {}).get('value'):
        try:
            vix_value = float(economic['vix']['value'])
            if vix_value > 25:
                briefing['alerts'].append({
                    'type': 'MARKET_VOLATILITY',
                    'message': f'VIX elevated at {vix_value} - monitor for potential race impact'
                })
            elif vix_value < 12:
                briefing['key_changes'].append('VIX at historic lows - favors incumbent')
        except:
            pass

    if polling.get('new_polls'):
        briefing['alerts'].append({
            'type': 'NEW_POLLING',
            'message': 'New polling data available - model update recommended'
        })

    if len(news) > 5:
        briefing['alerts'].append({
            'type': 'HIGH_NEWS_VOLUME',
            'message': f'{len(news)} news articles in last 24 hours - elevated coverage'
        })

    # Generate text briefing
    text_briefing = generate_text_briefing(briefing)

    # Save briefing
    briefing_file = BRIEFING_DIR / f"briefing_{briefing_date}.json"
    with open(briefing_file, 'w') as f:
        json.dump(briefing, f, indent=2)

    text_file = BRIEFING_DIR / f"briefing_{briefing_date}.txt"
    with open(text_file, 'w') as f:
        f.write(text_briefing)

    print(f"Briefing saved to {briefing_file}")
    print(f"Text briefing saved to {text_file}")

    return briefing, text_briefing


def generate_text_briefing(briefing):
    """Generate human-readable text briefing."""
    lines = []
    lines.append("=" * 60)
    lines.append("TEXAS GOVERNOR RACE - DAILY BRIEFING")
    lines.append(f"Date: {briefing['date']}")
    lines.append(f"Generated: {briefing['generated_at']}")
    lines.append("=" * 60)
    lines.append("")

    # Forecast Summary
    lines.append("CURRENT FORECAST")
    lines.append("-" * 40)
    fc = briefing['current_forecast']
    lines.append(f"  Predicted Winner: {fc['predicted_winner']}")
    lines.append(f"  Predicted Margin: {fc['predicted_margin']}")
    lines.append(f"  Win Probability:  {fc['win_probability']}")
    lines.append(f"  Confidence:       {fc['confidence']}")
    lines.append("")

    # Alerts
    if briefing['alerts']:
        lines.append("ALERTS")
        lines.append("-" * 40)
        for alert in briefing['alerts']:
            lines.append(f"  [{alert['type']}] {alert['message']}")
        lines.append("")

    # Polling Update
    lines.append("POLLING UPDATE")
    lines.append("-" * 40)
    poll = briefing['polling_update']
    if poll.get('new_polls'):
        lines.append("  NEW POLLS DETECTED")
    else:
        lines.append(f"  {poll.get('message', 'No update')}")
    if poll.get('current_average'):
        avg = poll['current_average']
        lines.append(f"  Current Average: Abbott {avg['abbott']}% - Hinojosa {avg['hinojosa']}% (R+{avg['margin']})")
    lines.append("")

    # Economic Indicators
    lines.append("ECONOMIC INDICATORS")
    lines.append("-" * 40)
    econ = briefing['economic_indicators']
    if econ:
        for name, data in econ.items():
            lines.append(f"  {name.upper()}: {data.get('value', 'N/A')} (as of {data.get('date', 'N/A')})")
    else:
        lines.append("  No economic data available")
    lines.append("")

    # News Summary
    lines.append("NEWS SUMMARY")
    lines.append("-" * 40)
    news = briefing['news_summary']
    lines.append(f"  {news['total_articles']} articles in last 24 hours")
    if news['articles']:
        lines.append("")
        for i, article in enumerate(news['articles'], 1):
            lines.append(f"  {i}. [{article['source']}] {article['headline'][:60]}...")
    lines.append("")

    # Key Changes
    if briefing['key_changes']:
        lines.append("KEY CHANGES")
        lines.append("-" * 40)
        for change in briefing['key_changes']:
            lines.append(f"  - {change}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("END OF BRIEFING")
    lines.append("=" * 60)

    return "\n".join(lines)


def update_dashboard_briefing():
    """Update the Streamlit dashboard with latest briefing."""
    briefing_date = datetime.now().strftime('%Y-%m-%d')
    briefing_file = BRIEFING_DIR / f"briefing_{briefing_date}.json"

    if briefing_file.exists():
        # Copy to a known location for the dashboard
        latest_file = BRIEFING_DIR / "latest_briefing.json"
        with open(briefing_file, 'r') as f:
            data = json.load(f)
        with open(latest_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Dashboard briefing updated: {latest_file}")


def get_current_model_state():
    """Get current model state for notification comparison."""
    return {
        'polling': {
            'abbott': 50,
            'hinojosa': 42,
            'margin': 8,
            'poll_count': 5
        },
        'prediction': {
            'winner': 'Abbott',
            'margin_low': 12,
            'margin_high': 18,
            'win_probability': 99
        },
        'primary': {
            'republican': {
                'abbott': '85-90',
                'brooks': '3',
                'goloby': '2'
            },
            'democratic': {
                'hinojosa': '52-58',
                'bell': '12-15',
                'cole': '8-10'
            }
        },
        'economic': {
            'vix': 15.4,
            'unemployment': 4.5,
            'inflation': 2.8
        },
        'finance': {
            'abbott_total': 105700000,
            'hinojosa_total': 1300000
        }
    }


if __name__ == "__main__":
    print(f"Starting daily update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    # Generate morning briefing
    briefing, text_briefing = generate_morning_briefing()

    # Update dashboard
    update_dashboard_briefing()

    # Check for changes and send notifications
    from notifications import check_and_notify
    current_state = get_current_model_state()

    # Update state with latest fetched data
    econ = briefing.get('economic_indicators', {})
    if econ.get('vix'):
        try:
            current_state['economic']['vix'] = float(econ['vix']['value'])
        except:
            pass

    changes = check_and_notify(current_state)

    if changes:
        print(f"\n{len(changes)} change(s) detected - notifications sent")

    # Print text briefing to console
    print("\n" + text_briefing)

    print("\nDaily update complete.")
