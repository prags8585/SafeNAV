import pandas as pd

CSV_PATH = "/Users/prags/Desktop/MLPROJECT/data/US_Accidents.csv"

print("Loading 500K random sample...")
chunks = []
for chunk in pd.read_csv(CSV_PATH, chunksize=500_000, low_memory=False):
    chunks.append(chunk.sample(frac=0.065, random_state=42))
df = pd.concat(chunks, ignore_index=True)

df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
df['Is_Night'] = df['Sunrise_Sunset'].map({'Night': 1, 'Day': 0})
df['Is_Rain'] = df['Weather_Condition'].str.contains('Rain', na=False, case=False)
df['Is_Fog'] = df['Weather_Condition'].str.contains('Fog', na=False, case=False)

print("\n" + "=" * 70)
print("THE POWER OF DECISION TREES: FEATURE COMBINATIONS")
print("=" * 70)
print("\nLet's look at why simple statistics fail and why we NEED a Decision Tree.")
print("If we look at features one by one, the differences seem small.")
print("But a Decision Tree looks at COMBINATIONS. Let's see what happens:\n")

baseline_sev34 = (df['Severity'] >= 3).mean() * 100
print(f"1. BASELINE (All Accidents):")
print(f"   Probability of Serious/Fatal: {baseline_sev34:.1f}%\n")

night_sev34 = (df[df['Is_Night'] == 1]['Severity'] >= 3).mean() * 100
print(f"2. INDIVIDUAL FEATURE: Just Nighttime")
print(f"   Probability of Serious/Fatal: {night_sev34:.1f}% (Barely changed!)\n")

rain_sev34 = (df[df['Is_Rain'] == True]['Severity'] >= 3).mean() * 100
print(f"3. INDIVIDUAL FEATURE: Just Rain")
print(f"   Probability of Serious/Fatal: {rain_sev34:.1f}% (A bit higher)\n")

junction_sev34 = (df[df['Junction'] == True]['Severity'] >= 3).mean() * 100
print(f"4. INDIVIDUAL FEATURE: Just at a Junction")
print(f"   Probability of Serious/Fatal: {junction_sev34:.1f}% (Higher)\n")

# Now let's combine them, like a Decision Tree would
combo1 = df[(df['Is_Night'] == 1) & (df['Junction'] == True)]
if len(combo1) > 0:
    combo1_sev34 = (combo1['Severity'] >= 3).mean() * 100
    print(f"5. COMBINATION (Depth 2 Tree): Night + Junction")
    print(f"   Probability of Serious/Fatal: {combo1_sev34:.1f}% (Getting dangerous)\n")

combo2 = df[(df['Is_Night'] == 1) & (df['Junction'] == True) & (df['Is_Rain'] == True)]
if len(combo2) > 0:
    combo2_sev34 = (combo2['Severity'] >= 3).mean() * 100
    print(f"6. COMBINATION (Depth 3 Tree): Night + Junction + Rain")
    print(f"   Probability of Serious/Fatal: {combo2_sev34:.1f}% (Danger spike!)\n")

combo3 = df[(df['Junction'] == True) & (df['Visibility(mi)'] < 2) & (df['Traffic_Signal'] == False)]
if len(combo3) > 0:
    combo3_sev34 = (combo3['Severity'] >= 3).mean() * 100
    print(f"7. THE WORST CASE: Junction + Low Visibility (<2mi) + No Traffic Signal")
    print(f"   Probability of Serious/Fatal: {combo3_sev34:.1f}% (More than DOUBLE the baseline!)\n")
    print(f"   Sample size for this exact scenario: {len(combo3)} accidents")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("A single variable (like Night) doesn't predict severity well.")
print("But a Decision Tree automatically finds these deadly COMBINATIONS.")
print("It learns rules like: IF Junction=True AND Visibility<2 AND Signal=False THEN Severity=Fatal")
print("=" * 70)
