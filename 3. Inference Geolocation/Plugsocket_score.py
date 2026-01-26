import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========== INPUT FILES ==========
results_csv = "results_traffickam.csv"          
mapping_csv = "plug_type_with_country_mapping.csv"  

# ========== LOAD DATA ==========
df = pd.read_csv(results_csv)
mapping_df = pd.read_csv(mapping_csv)

# --- Standardize column names ---
mapping_df.columns = mapping_df.columns.str.strip().str.lower()

# --- Normalize country text function ---
def normalize_text(value):
    if pd.isna(value):
        return None
    val = str(value).strip().lower()
    aliases = {
        "united states": "usa",
        "united states of america": "usa",
        "america": "usa",
        "united_states": "usa",
        "uk": "united kingdom",
        "united kingdom": "united kingdom",
    }
    return aliases.get(val, val)

# --- Apply normalization ---
df["country_name"] = df["country_name"].apply(normalize_text)
mapping_df["country"] = mapping_df["country"].apply(normalize_text)

# --- Normalize socket/plug text ---
df["socket_class"] = df["socket_class"].astype(str).str.strip().str.upper()
mapping_df["plug type"] = mapping_df["plug type"].str.strip().str.upper()

# --- Build valid plug-country pairs ---
valid_pairs = set(zip(mapping_df["plug type"], mapping_df["country"]))

# ========== ASSIGN SCORES ==========
def assign_score(row):
    if row["socket_detected"].strip().upper() == "NO":
        return None  # leave empty if socket not detected
    if row["socket_class"] == "NOISE":
        return 0   # neutral
    if (row["socket_class"], row["country_name"]) in valid_pairs:
        return 1   # correct
    return -1      # wrong

df["socket_score"] = df.apply(assign_score, axis=1)

# ========== SOCKET_DETECTED YES SUMMARY ==========
df_yes = df[df["socket_detected"].str.strip().str.upper() == "YES"]
total_yes = len(df_yes)
correct_count = (df_yes["socket_score"] == 1).sum()
wrong_count = (df_yes["socket_score"] == -1).sum()
noise_count = (df_yes["socket_score"] == 0).sum()

print("\n=== SOCKET DETECTED = YES SUMMARY ===")
print(f"Total YES: {total_yes}")
print(f"Correct: {correct_count}")
print(f"Wrong: {wrong_count}")
print(f"Noise (neutral): {noise_count}")

# ========== CONFIDENCE THRESHOLD PERFORMANCE ==========
class_thresholds = [70, 80, 90]
threshold_results = []

# Baseline (no threshold)
df_non_neutral = df_yes[df_yes["socket_score"].notna() & (df_yes["socket_score"] != 0)].copy()
correct = (df_non_neutral["socket_score"] == 1).sum()
wrong = (df_non_neutral["socket_score"] == -1).sum()
accuracy = (correct / (correct + wrong) * 100) if (correct + wrong) > 0 else 0
threshold_results.append(("No Threshold", correct, wrong, accuracy))

# Threshold-based
for thr in class_thresholds:
    df_thr = df_yes[(df_yes["class_confidence"] >= thr) & df_yes["socket_score"].notna() & (df_yes["socket_score"] != 0)].copy()
    df_thr = df_thr.loc[df_thr.groupby("image_id")["class_confidence"].idxmax()]  # pick max confidence per image
    correct = (df_thr["socket_score"] == 1).sum()
    wrong = (df_thr["socket_score"] == -1).sum()
    accuracy = (correct / (correct + wrong) * 100) if (correct + wrong) > 0 else 0
    threshold_results.append((f">{thr}%", correct, wrong, accuracy))

# Print summary like in your description
print("\n=== CONFIDENCE THRESHOLD PERFORMANCE ===")
for label, correct, wrong, acc in threshold_results:
    print(f"{label}: Correct={correct}, Wrong={wrong}, Accuracy={acc:.2f}%")

# ========== BAR CHART ==========
# ========== BAR CHART ==========
labels = [r[0] for r in threshold_results]
accuracies = [r[3] for r in threshold_results]

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(labels)))  # nice gradient palette

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, accuracies, color=colors, edgecolor="black", linewidth=1.2)

# Title and labels
plt.title("Socket Classification Accuracy at Different Confidence Thresholds", fontsize=14, weight="bold")
plt.ylabel("Accuracy (%)", fontsize=12)

# Add some headroom above highest bar for padding (5px equivalent in y-scale)
max_acc = max(accuracies)
plt.ylim(0, max_acc + 5)

plt.grid(axis="y", linestyle="--", alpha=0.6)

# Annotate values on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{acc:.2f}%", ha="center", va="bottom", fontsize=10, weight="bold")

plt.tight_layout()
plt.savefig("socket_accuracy_chart.png", dpi=300, bbox_inches="tight")
plt.show()

# ========== FINAL SUMMARY ==========
print("\n=== FINAL SUMMARY ===")
print(f"Total YES: {total_yes}")
print(f"Noise (neutral): {noise_count}")
print(f"Considered for analysis: {total_yes - noise_count} (Correct + Wrong)")

# ========== SAVE UPDATED FILE ==========
output_csv = "results_traffickam_with_scores.csv"
df.to_csv(output_csv, index=False)
print(f"\n✅ Updated file with 'socket_score' column saved as {output_csv}")
