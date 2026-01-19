#!/usr/bin/env python3
"""
Notification System for Texas Governor Race Analysis
Sends alerts when significant changes are detected.
"""

import os
import json
import requests
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Pushover Configuration
PUSHOVER_USER_KEY = os.getenv('PUSHOVER_USER_KEY', '')
PUSHOVER_APP_TOKEN = os.getenv('PUSHOVER_APP_TOKEN', '')
NOTIFICATION_EMAIL = os.getenv('NOTIFICATION_EMAIL', '')

DATA_DIR = Path(__file__).parent / 'data'
STATE_FILE = Path(__file__).parent / 'model_state.json'


def load_previous_state():
    """Load the previous model state for comparison."""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return None


def save_current_state(state):
    """Save current state for future comparison."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def detect_changes(previous, current):
    """Detect any changes between previous and current state."""
    changes = []

    if previous is None:
        return [{'type': 'INITIAL', 'message': 'Initial state recorded', 'severity': 'info'}]

    # Check polling changes
    prev_margin = previous.get('polling', {}).get('margin', 0)
    curr_margin = current.get('polling', {}).get('margin', 0)
    margin_change = curr_margin - prev_margin

    if abs(margin_change) >= 0.1:  # Any change >= 0.1 points
        severity = 'high' if abs(margin_change) >= 2 else 'medium' if abs(margin_change) >= 1 else 'low'
        direction = 'tightened' if margin_change < 0 else 'widened'
        changes.append({
            'type': 'POLLING_SHIFT',
            'message': f'Polling margin {direction} by {abs(margin_change):.1f} points (now R+{curr_margin})',
            'severity': severity,
            'previous': prev_margin,
            'current': curr_margin
        })

    # Check prediction changes
    prev_pred = previous.get('prediction', {}).get('margin_low', 0)
    curr_pred = current.get('prediction', {}).get('margin_low', 0)
    pred_change = curr_pred - prev_pred

    if abs(pred_change) >= 0.5:
        severity = 'high' if abs(pred_change) >= 3 else 'medium'
        changes.append({
            'type': 'PREDICTION_CHANGE',
            'message': f'Model prediction shifted by {abs(pred_change):.1f} points',
            'severity': severity,
            'previous': prev_pred,
            'current': curr_pred
        })

    # Check win probability changes
    prev_prob = previous.get('prediction', {}).get('win_probability', 0)
    curr_prob = current.get('prediction', {}).get('win_probability', 0)
    prob_change = curr_prob - prev_prob

    if abs(prob_change) >= 0.1:  # Any change >= 0.1%
        severity = 'high' if abs(prob_change) >= 5 else 'medium' if abs(prob_change) >= 1 else 'low'
        changes.append({
            'type': 'PROBABILITY_CHANGE',
            'message': f'Win probability changed from {prev_prob}% to {curr_prob}%',
            'severity': severity,
            'previous': prev_prob,
            'current': curr_prob
        })

    # Check VIX/market volatility
    prev_vix = previous.get('economic', {}).get('vix', 0)
    curr_vix = current.get('economic', {}).get('vix', 0)
    vix_change = curr_vix - prev_vix

    if abs(vix_change) >= 2:
        severity = 'high' if abs(vix_change) >= 5 else 'medium'
        direction = 'increased' if vix_change > 0 else 'decreased'
        changes.append({
            'type': 'MARKET_VOLATILITY',
            'message': f'VIX {direction} by {abs(vix_change):.1f} points (now {curr_vix})',
            'severity': severity,
            'previous': prev_vix,
            'current': curr_vix
        })

    # Check for new polls
    prev_poll_count = previous.get('polling', {}).get('poll_count', 0)
    curr_poll_count = current.get('polling', {}).get('poll_count', 0)

    if curr_poll_count > prev_poll_count:
        new_polls = curr_poll_count - prev_poll_count
        changes.append({
            'type': 'NEW_POLLS',
            'message': f'{new_polls} new poll(s) added to the model',
            'severity': 'medium',
            'new_count': new_polls
        })

    # Check fundraising changes
    prev_abbott_funds = previous.get('finance', {}).get('abbott_total', 0)
    curr_abbott_funds = current.get('finance', {}).get('abbott_total', 0)
    prev_hinojosa_funds = previous.get('finance', {}).get('hinojosa_total', 0)
    curr_hinojosa_funds = current.get('finance', {}).get('hinojosa_total', 0)

    abbott_change = curr_abbott_funds - prev_abbott_funds
    hinojosa_change = curr_hinojosa_funds - prev_hinojosa_funds

    if abbott_change >= 1000000:  # $1M+ change
        changes.append({
            'type': 'FUNDRAISING_UPDATE',
            'message': f'Abbott raised additional ${abbott_change/1000000:.1f}M (total: ${curr_abbott_funds/1000000:.1f}M)',
            'severity': 'low',
            'candidate': 'Abbott'
        })

    if hinojosa_change >= 100000:  # $100K+ change (lower threshold for underdog)
        changes.append({
            'type': 'FUNDRAISING_UPDATE',
            'message': f'Hinojosa raised additional ${hinojosa_change/1000:.0f}K (total: ${curr_hinojosa_funds/1000000:.2f}M)',
            'severity': 'medium',
            'candidate': 'Hinojosa'
        })

    return changes


def send_pushover_notification(changes, current_state):
    """Send Pushover notification about detected changes."""
    if not PUSHOVER_USER_KEY or not PUSHOVER_APP_TOKEN:
        print("Pushover not configured - skipping notification")
        return False

    high_severity = [c for c in changes if c.get('severity') == 'high']
    is_urgent = len(high_severity) > 0

    # Build notification title
    title = f"{'URGENT: ' if is_urgent else ''}TX Governor Race Alert"

    # Build message body
    lines = [f"{len(changes)} change(s) detected:\n"]

    for change in changes:
        severity_marker = "!!!" if change['severity'] == 'high' else "!" if change['severity'] == 'medium' else "-"
        lines.append(f"{severity_marker} {change['type']}: {change['message']}")

    lines.append("")

    # Current state
    pred = current_state.get('prediction', {})
    poll = current_state.get('polling', {})

    lines.append(f"Forecast: Abbott (R) wins R+{pred.get('margin_low', 12)}-{pred.get('margin_high', 18)}%")
    lines.append(f"Polling: Abbott {poll.get('abbott', 50)}% - Hinojosa {poll.get('hinojosa', 42)}%")

    message = "\n".join(lines)

    # Set priority: 1 for high priority (urgent), 0 for normal
    priority = 1 if is_urgent else 0

    try:
        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": PUSHOVER_APP_TOKEN,
                "user": PUSHOVER_USER_KEY,
                "title": title,
                "message": message,
                "priority": priority,
                "url": "http://207.254.38.26:8502",
                "url_title": "View Dashboard"
            },
            timeout=30
        )

        result = response.json()
        if result.get('status') == 1:
            print(f"Pushover notification sent successfully")
            return True
        else:
            print(f"Pushover API error: {result.get('errors', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"Failed to send Pushover notification: {e}")
        return False


def save_notification_log(changes, current_state):
    """Save notification to local log file."""
    log_dir = Path(__file__).parent / 'notifications'
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"notification_{timestamp}.json"

    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'changes': changes,
        'current_state': current_state
    }

    with open(log_file, 'w') as f:
        json.dump(log_entry, f, indent=2)

    # Also append to master log
    master_log = log_dir / "notification_history.jsonl"
    with open(master_log, 'a') as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"Notification logged to {log_file}")


def check_and_notify(current_state):
    """Main function to check for changes and send notifications."""
    previous_state = load_previous_state()
    changes = detect_changes(previous_state, current_state)

    if changes:
        print(f"\n{len(changes)} change(s) detected:")
        for change in changes:
            print(f"  [{change['severity'].upper()}] {change['type']}: {change['message']}")

        # Log all notifications
        save_notification_log(changes, current_state)

        # Send Pushover notification for any changes
        send_pushover_notification(changes, current_state)

    else:
        print("No changes detected")

    # Save current state for next comparison
    save_current_state(current_state)

    return changes


if __name__ == "__main__":
    # Test with sample state
    test_state = {
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

    print("Running notification check...")
    changes = check_and_notify(test_state)
    print(f"\nTotal changes: {len(changes)}")
