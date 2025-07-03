# Bitcoin Analysis Bot - Alert Logic Documentation

## Alert Frequency and Types

### 1. Data Collection
- **Frequency**: Every 10 minutes (UPDATE_INTERVAL_MINUTES=10)
- **Sources**: Twitter (25 posts), Reddit (20 posts + 10 comments), Price data

### 2. Alert Types

#### Critical Alerts 🚨
**Triggered immediately when:**
- Price change > 5% in 24h prediction
- Sentiment score < -0.7 or > 0.7 (very strong sentiment)
- Technical analysis shows STRONG_BUY or STRONG_SELL

**Timing:**
- Sent immediately when conditions are met
- Minimum 2-minute gap between critical alerts (anti-spam)

#### Regular Updates 📊
**Triggered when:**
- No alert sent in 30 minutes (forced update)
- Price change > 2% (but < 5%)
- Moderate sentiment change (threshold based on confidence)
- Any technical signal (BUY, SELL, STRONG_BUY, STRONG_SELL)

**Timing:**
- Minimum 10-minute gap between regular alerts
- Aligns with data collection cycle

#### Hourly Summaries 📈
**Content:**
- Current price
- Recent sentiment (last 2 hours)
- Number of posts analyzed
- System status

**Timing:**
- Every 2 hours to avoid spam

#### Startup Message 🤖
**Content:**
- Bot status and configuration
- Alert types explanation
- System capabilities

**Timing:**
- Once per day (tracked via last_startup.txt)

### 3. Alert Priority
1. **Critical alerts** - Highest priority, sent immediately
2. **Regular updates** - Sent when significant but non-critical changes occur
3. **Hourly summaries** - Background status updates
4. **Startup message** - Daily system notification

### 4. Anti-Spam Protection
- Critical alerts: 2-minute minimum gap
- Regular alerts: 10-minute minimum gap
- Startup message: Once per day
- Hourly summary: Every 2 hours

### 5. Data Requirements
- Alerts are only sent if sufficient data is collected
- Minimum posts required for sentiment analysis
- Price data must be available for technical analysis

## Alert Message Format

### Critical Alert 🚨
```
🚨 **CRITICAL BITCOIN ALERT** 🚨

💰 **Price:** $X,XXX.XX 🔴/🟢
📈 **24h Change:** ±X.XX%
💭 **Sentiment:** X.XXX (Description) 😊
📊 **Posts:** XXX (Conf: X.XX)
📈 **Technical:** STRONG_BUY 🚀
🎯 **Strength:** X.XX/1.0
🔮 **24h Prediction:** $X,XXX.XX 🔺
🎯 **Expected Change:** ±X.XX%
🎓 **Confidence:** X.XX

⚠️ **This is a critical alert - significant market movement detected!**

⏰ HH:MM:SS UTC, YYYY-MM-DD
```

### Regular Update 🔮
```
🔮 **Bitcoin Analysis Update** 🔮

💰 **Price:** $X,XXX.XX 🟢
💭 **Sentiment:** X.XXX (Positive) 😊
📊 **Posts:** XXX (Conf: X.XX)
📈 **Technical:** BUY 📈
🔮 **24h Prediction:** $X,XXX.XX 🔺
🎯 **Expected Change:** +X.XX%
🎓 **Confidence:** X.XX

⏰ HH:MM:SS UTC, YYYY-MM-DD
```

## Configuration
- All thresholds can be adjusted in the code
- Environment variables control timing
- Telegram credentials required for alerts
- Bot works without alerts if Telegram not configured
