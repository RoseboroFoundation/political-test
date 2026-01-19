#!/bin/bash
# Setup daily update scheduler for Texas Governor Race Analysis

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_PATH="$SCRIPT_DIR/venv/bin/python"
UPDATE_SCRIPT="$SCRIPT_DIR/daily_update.py"
LOG_FILE="$SCRIPT_DIR/logs/daily_update.log"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Create the cron job entry (runs at 6:00 AM daily)
CRON_JOB="0 6 * * * cd $SCRIPT_DIR && $PYTHON_PATH $UPDATE_SCRIPT >> $LOG_FILE 2>&1"

echo "Setting up daily update scheduler..."
echo ""
echo "Cron job to add:"
echo "$CRON_JOB"
echo ""

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "daily_update.py"; then
    echo "Cron job already exists. Skipping."
else
    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "Cron job added successfully!"
fi

echo ""
echo "To manually run the daily update:"
echo "  cd $SCRIPT_DIR && $PYTHON_PATH $UPDATE_SCRIPT"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To remove the cron job:"
echo "  crontab -e  (and delete the line)"
