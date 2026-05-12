import yfinance as yf
import pandas as pd

# Enable debug mode to see what's happening
yf.enable_debug_mode()

# Try a simple download
print("🔍 Testing yfinance download...")
df = yf.download("BTC-USD", start="2025-11-01", end="2025-12-01", progress=True)

print(f"✅ Rows downloaded: {len(df)}")
print(f"✅ Columns: {list(df.columns) if not df.empty else 'EMPTY'}")