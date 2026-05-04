"""
What does the data ACTUALLY say?
Does severity correlate with weather, time, road features?
Let's find out from a 500K sample of the real data.
"""
import pandas as pd
import numpy as np

CSV_PATH = "/Users/prags/Desktop/MLPROJECT/data/US_Accidents.csv"

print("Loading 500K random sample...")
# Read all rows but sample 500K to keep it fast
chunks = []
for chunk in pd.read_csv(CSV_PATH, chunksize=500_000, low_memory=False):
    chunks.append(chunk.sample(frac=0.065, random_state=42))  # ~32.5K per chunk × ~15 chunks ≈ 500K
df = pd.concat(chunks, ignore_index=True)
print(f"Sample size: {len(df):,} rows\n")

# Extract time features
df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.dayofweek  # 0=Mon, 6=Sun
df['Is_Night'] = df['Sunrise_Sunset'].map({'Night': 1, 'Day': 0})

# ============================================================
print("=" * 70)
print("QUESTION 1: Does SEVERITY actually vary by conditions?")
print("          (Or is it just random noise?)")
print("=" * 70)

# ── Severity distribution in our sample ──
print("\n📊 Severity Distribution:")
sev_counts = df['Severity'].value_counts().sort_index()
for sev, cnt in sev_counts.items():
    pct = cnt / len(df) * 100
    print(f"  Severity {sev}: {cnt:>8,} ({pct:>5.1f}%)")

# ============================================================
print("\n" + "=" * 70)
print("QUESTION 2: Are Night accidents more SEVERE than Day?")
print("=" * 70)

day_night = df.groupby(['Sunrise_Sunset', 'Severity']).size().unstack(fill_value=0)
day_night_pct = day_night.div(day_night.sum(axis=1), axis=0) * 100

print("\n  Time     | Sev 1  | Sev 2   | Sev 3   | Sev 4")
print("  " + "-" * 50)
for period in ['Day', 'Night']:
    if period in day_night_pct.index:
        row = day_night_pct.loc[period]
        total = day_night.loc[period].sum()
        print(f"  {period:8s} | {row.get(1,0):5.1f}% | {row.get(2,0):5.1f}%  | {row.get(3,0):5.1f}%  | {row.get(4,0):5.1f}%   (n={total:,})")

# ============================================================
print("\n" + "=" * 70)
print("QUESTION 3: Does LOW VISIBILITY mean higher severity?")
print("=" * 70)

vis_bins = pd.cut(df['Visibility(mi)'], bins=[0, 1, 3, 5, 10, 100], labels=['<1mi', '1-3mi', '3-5mi', '5-10mi', '>10mi'])
vis_sev = df.groupby([vis_bins, 'Severity']).size().unstack(fill_value=0)
vis_sev_pct = vis_sev.div(vis_sev.sum(axis=1), axis=0) * 100

print(f"\n  Visibility | Sev 1  | Sev 2   | Sev 3   | Sev 4  | Avg Severity")
print("  " + "-" * 65)
for vis_range in ['<1mi', '1-3mi', '3-5mi', '5-10mi', '>10mi']:
    if vis_range in vis_sev_pct.index:
        row = vis_sev_pct.loc[vis_range]
        n = vis_sev.loc[vis_range].sum()
        # Calculate avg severity for this bin
        avg_sev = df[vis_bins == vis_range]['Severity'].mean()
        print(f"  {vis_range:10s}  | {row.get(1,0):5.1f}% | {row.get(2,0):5.1f}%  | {row.get(3,0):5.1f}%  | {row.get(4,0):5.1f}% | {avg_sev:.3f}   (n={n:,})")

# ============================================================
print("\n" + "=" * 70)
print("QUESTION 4: Does TEMPERATURE affect severity?")
print("=" * 70)

temp_bins = pd.cut(df['Temperature(F)'].dropna(), bins=[-20, 32, 50, 70, 90, 150], 
                   labels=['<32°F (Freezing)', '32-50°F', '50-70°F', '70-90°F', '>90°F'])
temp_df = df.dropna(subset=['Temperature(F)'])
temp_df_binned = pd.cut(temp_df['Temperature(F)'], bins=[-20, 32, 50, 70, 90, 150], 
                        labels=['<32°F (Freezing)', '32-50°F', '50-70°F', '70-90°F', '>90°F'])

print(f"\n  Temperature      | Avg Severity | % Sev 3+4 (Serious+Fatal)")
print("  " + "-" * 60)
for temp_range in ['<32°F (Freezing)', '32-50°F', '50-70°F', '70-90°F', '>90°F']:
    mask = temp_df_binned == temp_range
    if mask.sum() > 0:
        avg_sev = temp_df.loc[mask, 'Severity'].mean()
        pct_severe = (temp_df.loc[mask, 'Severity'] >= 3).mean() * 100
        n = mask.sum()
        print(f"  {temp_range:18s} | {avg_sev:.3f}        | {pct_severe:5.1f}%                  (n={n:,})")

# ============================================================
print("\n" + "=" * 70)
print("QUESTION 5: Does TIME OF DAY affect severity?")
print("=" * 70)

hour_groups = {
    'Late Night (12-5AM)': (0, 5),
    'Morning Rush (6-9AM)': (6, 9),
    'Midday (10AM-3PM)': (10, 15),
    'Evening Rush (4-7PM)': (16, 19),
    'Night (8-11PM)': (20, 23)
}

print(f"\n  Time Period          | Count    | Avg Severity | % Sev 3+4")
print("  " + "-" * 65)
for label, (start, end) in hour_groups.items():
    mask = (df['Hour'] >= start) & (df['Hour'] <= end)
    if mask.sum() > 0:
        avg_sev = df.loc[mask, 'Severity'].mean()
        pct_severe = (df.loc[mask, 'Severity'] >= 3).mean() * 100
        n = mask.sum()
        print(f"  {label:22s} | {n:>7,} | {avg_sev:.3f}        | {pct_severe:5.1f}%")

# ============================================================
print("\n" + "=" * 70)
print("QUESTION 6: Do ROAD FEATURES affect severity?")
print("=" * 70)

road_features = ['Junction', 'Crossing', 'Traffic_Signal', 'Stop', 'Station', 'Railway']
print(f"\n  Feature          | Present Avg Sev | Absent Avg Sev | Difference | % Sev3+4 Present | % Sev3+4 Absent")
print("  " + "-" * 95)
for feat in road_features:
    if feat in df.columns:
        present = df[df[feat] == True]['Severity']
        absent = df[df[feat] == False]['Severity']
        if len(present) > 100:
            diff = present.mean() - absent.mean()
            direction = "↑ MORE severe" if diff > 0 else "↓ LESS severe"
            pct_present = (present >= 3).mean() * 100
            pct_absent = (absent >= 3).mean() * 100
            print(f"  {feat:18s} | {present.mean():.3f}           | {absent.mean():.3f}          | {diff:+.3f} {direction} | {pct_present:5.1f}%             | {pct_absent:5.1f}%")

# ============================================================
print("\n" + "=" * 70)
print("QUESTION 7: Does WEATHER CONDITION affect severity?")
print("=" * 70)

weather_sev = df.groupby('Weather_Condition')['Severity'].agg(['mean', 'count', lambda x: (x >= 3).mean() * 100])
weather_sev.columns = ['avg_severity', 'count', 'pct_severe']
weather_sev = weather_sev[weather_sev['count'] >= 200].sort_values('avg_severity', ascending=False)

print(f"\n  Weather Condition        | Count    | Avg Severity | % Sev 3+4")
print("  " + "-" * 70)
for weather, row in weather_sev.head(15).iterrows():
    print(f"  {str(weather):26s} | {int(row['count']):>7,} | {row['avg_severity']:.3f}        | {row['pct_severe']:5.1f}%")

# ============================================================
print("\n" + "=" * 70)
print("QUESTION 8: Which STATES have the most severe accidents?")
print("=" * 70)

state_sev = df.groupby('State')['Severity'].agg(['mean', 'count', lambda x: (x >= 3).mean() * 100])
state_sev.columns = ['avg_severity', 'count', 'pct_severe']
state_sev = state_sev[state_sev['count'] >= 500].sort_values('avg_severity', ascending=False)

print(f"\n  Top 10 States by Average Severity:")
print(f"  State | Count    | Avg Severity | % Sev 3+4")
print("  " + "-" * 50)
for state, row in state_sev.head(10).iterrows():
    print(f"  {state:5s} | {int(row['count']):>7,} | {row['avg_severity']:.3f}        | {row['pct_severe']:5.1f}%")

# ============================================================
print("\n" + "=" * 70)
print("QUESTION 9: HUMIDITY × SEVERITY — is there a relationship?")
print("=" * 70)

hum_bins = pd.cut(df['Humidity(%)'].dropna(), bins=[0, 30, 50, 70, 90, 100],
                  labels=['<30%', '30-50%', '50-70%', '70-90%', '>90%'])
hum_df = df.dropna(subset=['Humidity(%)'])
hum_df_binned = pd.cut(hum_df['Humidity(%)'], bins=[0, 30, 50, 70, 90, 100],
                       labels=['<30%', '30-50%', '50-70%', '70-90%', '>90%'])

print(f"\n  Humidity     | Avg Severity | % Sev 3+4")
print("  " + "-" * 45)
for hum_range in ['<30%', '30-50%', '50-70%', '70-90%', '>90%']:
    mask = hum_df_binned == hum_range
    if mask.sum() > 0:
        avg_sev = hum_df.loc[mask, 'Severity'].mean()
        pct_severe = (hum_df.loc[mask, 'Severity'] >= 3).mean() * 100
        n = mask.sum()
        print(f"  {hum_range:12s}  | {avg_sev:.3f}        | {pct_severe:5.1f}%   (n={n:,})")

# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY: Can ML learn from this data?")
print("=" * 70)
print("""
The answer is based on whether features VARY with the target (Severity).
If all the numbers above were identical across groups, ML would fail.
Let's check the spread of average severity across conditions:
""")

# Compute spread for each feature
spreads = {}
spreads['Visibility'] = df.groupby(pd.cut(df['Visibility(mi)'], bins=[0,1,3,5,10,100]))['Severity'].mean().max() - \
                         df.groupby(pd.cut(df['Visibility(mi)'], bins=[0,1,3,5,10,100]))['Severity'].mean().min()
spreads['Temperature'] = df.groupby(pd.cut(df['Temperature(F)'].dropna(), bins=[-20,32,50,70,90,150]))['Severity'].mean().max() - \
                          df.groupby(pd.cut(df['Temperature(F)'].dropna(), bins=[-20,32,50,70,90,150]))['Severity'].mean().min()
spreads['Hour'] = df.groupby(df['Hour'])['Severity'].mean().max() - df.groupby(df['Hour'])['Severity'].mean().min()
spreads['Day/Night'] = abs(df[df['Is_Night']==1]['Severity'].mean() - df[df['Is_Night']==0]['Severity'].mean())
spreads['Weather'] = df.groupby('Weather_Condition')['Severity'].mean().max() - df.groupby('Weather_Condition')['Severity'].mean().min()
spreads['Junction'] = abs(df[df['Junction']==True]['Severity'].mean() - df[df['Junction']==False]['Severity'].mean())

print(f"  Feature          | Severity Spread | Signal Strength")
print("  " + "-" * 55)
for feat, spread in sorted(spreads.items(), key=lambda x: -x[1]):
    if spread > 0.15:
        signal = "🟢 STRONG"
    elif spread > 0.05:
        signal = "🟡 MODERATE"
    else:
        signal = "🔴 WEAK"
    print(f"  {feat:18s} | {spread:.3f}           | {signal}")

print("""
🟢 STRONG = Feature clearly differentiates severity → useful for ML
🟡 MODERATE = Some signal → ML can use it in combination with others
🔴 WEAK = Feature alone doesn't predict severity well → may still help in combination
""")
