# make_figs.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from calendar import month_abbr

DATA_CSV = "hengsha_hourly_2020_2024.csv"

df_raw = pd.read_csv(DATA_CSV)

required_cols = {"YEAR", "MO", "DY", "HR", "ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS10M"}
missing = required_cols - set(df_raw.columns)
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

df = df_raw.copy()
df["timestamp"] = pd.to_datetime(
    dict(year=df["YEAR"], month=df["MO"], day=df["DY"], hour=df["HR"]), errors="coerce"
)
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp")

df = df.rename(
    columns={
        "ALLSKY_SFC_SW_DWN": "ghi",
        "T2M": "temp_c",
        "RH2M": "rh_pct",
        "WS10M": "wind_ms",
    }
)

df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["hour"] = df["timestamp"].dt.hour
df["doy"] = df["timestamp"].dt.dayofyear

out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)

# 1) Train/val/test timeline strip
splits = [
    ("Train 2020-2022", "2020-01-01", "2022-12-31", "#6baed6"),
    ("Val 2023", "2023-01-01", "2023-12-31", "#fd8d3c"),
    ("Test 2024", "2024-01-01", "2024-12-31", "#74c476"),
]
fig, ax = plt.subplots(figsize=(10, 1.2))
y = 0.5
for label, start, end, color in splits:
    ax.plot(pd.to_datetime([start, end]), [y, y], lw=20, solid_capstyle="butt", color=color, alpha=0.85)
    ax.text(pd.to_datetime(start) + (pd.to_datetime(end) - pd.to_datetime(start)) / 2, y + 0.05, label,
            ha="center", va="bottom", fontsize=9)
ax.set_xlim(df["timestamp"].min(), df["timestamp"].max())
ax.set_yticks([])
ax.set_xlabel("Chronological split")
sns.despine(left=True, bottom=False)
fig.tight_layout()
fig.savefig(out_dir / "split_timeline.png", dpi=300)

# 1b) Monthly day-range split (stacked, per month)
day_bins = {
    "Prediction (Day 20-End)": (21, 31),
    "Testing (Day 16-20)": (16, 20),
    "Validation (Day 11-15)": (11, 15),
    "Training (Day 1-10)": (1, 10),
}
colors = {
    "Training (Day 1-10)": "#1b9e77",
    "Validation (Day 11-15)": "#fdae6b",
    "Testing (Day 16-20)": "#6a51a3",
    "Prediction (Day 20-End)": "#3182bd",
}

month_order = list(range(1, 13))
month_labels = [month_abbr[m] for m in month_order]

counts = {label: [] for label in day_bins}
for m in month_order:
    df_m = df[df["month"] == m]
    for label, (lo, hi) in day_bins.items():
        counts[label].append(((df_m["DY"] >= lo) & (df_m["DY"] <= hi)).sum())

fig, ax = plt.subplots(figsize=(8.5, 6))
left = np.zeros(len(month_order))
for label in ["Training (Day 1-10)", "Validation (Day 11-15)", "Testing (Day 16-20)", "Prediction (Day 20-End)"]:
    ax.barh(month_labels, counts[label], left=left, color=colors[label], label=label, height=0.6)
    left += np.array(counts[label])

ax.set_xlabel("Number of records")
ax.set_ylabel("Month")
ax.set_title("Monthly dataset split by day ranges")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)
fig.tight_layout()
fig.savefig(out_dir / "monthly_day_split.png", dpi=300)

# 2) Seasonal + diurnal climatology (4-panel)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
monthly = df.groupby("month")["ghi"].mean()
axes[0, 0].plot(monthly.index, monthly.values, marker="o")
axes[0, 0].set_title("Monthly mean GHI")
axes[0, 0].set_xlabel("Month"); axes[0, 0].set_ylabel("GHI (W/m^2)")

daily = df.groupby("month")["temp_c"].mean()
axes[0, 1].plot(daily.index, daily.values, marker="o", color="darkorange")
axes[0, 1].set_title("Monthly mean air temperature")
axes[0, 1].set_xlabel("Month"); axes[0, 1].set_ylabel("Temp (C)")

for col, ax, title, color in [
    ("ghi", axes[1, 0], "Diurnal GHI percentiles", "#3182bd"),
    ("temp_c", axes[1, 1], "Diurnal temperature percentiles", "#d95f02"),
]:
    grp = df.groupby("hour")[col]
    h = grp.quantile([0.1, 0.5, 0.9]).unstack()
    ax.fill_between(h.index, h[0.1], h[0.9], color=color, alpha=0.2, label="P10–P90")
    ax.plot(h.index, h[0.5], color=color, lw=2, label="Median")
    ax.set_xlabel("Hour"); ax.set_title(title)
    ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out_dir / "seasonal_diurnal.png", dpi=300)

# 3) Feature relationships
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
hb = ax[0].hexbin(df["ghi"], df["temp_c"], gridsize=40, cmap="viridis", mincnt=5)
ax[0].set_xlabel("GHI (W/m^2)"); ax[0].set_ylabel("Air temperature (C)")
ax[0].set_title("GHI vs temperature density")
fig.colorbar(hb, ax=ax[0], label="Count")

wind_bins = pd.qcut(df["wind_ms"], q=25, duplicates="drop")
wind_mean = df.groupby(wind_bins, observed=False)["ghi"].mean()
wind_centers = wind_mean.index.map(lambda x: getattr(x, "mid", np.nan))
ax[1].plot(wind_centers, wind_mean.values, marker="o", color="#31a354")
ax[1].set_xlabel("Wind speed (m/s)"); ax[1].set_ylabel("Mean GHI (W/m^2)")
ax[1].set_title("GHI vs wind speed (binned)")
fig.tight_layout()
fig.savefig(out_dir / "feature_relationships.png", dpi=300)

print("Saved figures to", out_dir.resolve())