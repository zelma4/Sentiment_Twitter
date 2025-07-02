#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Create database
python -c "from database.models import create_database; create_database()"

# Run initial data collection (if APIs are configured)
python -c "
try:
    from data_collectors.price_collector import PriceCollector
    collector = PriceCollector()
    collector.collect_and_save(include_historical=True, historical_days=7)
    print('Initial price data collected')
except Exception as e:
    print(f'Initial data collection failed: {e}')
"
