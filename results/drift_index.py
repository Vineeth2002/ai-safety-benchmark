import matplotlib.pyplot as plt
import numpy as np

# Dummy data for 8 days (replace with your real JSD values if available)
dates = [f"Day {i}" for i in range(1, 9)]
jsd = np.random.uniform(0, 0.2, size=8)  # Random JSD values

plt.figure(figsize=(8,4))
plt.plot(dates, jsd, marker='o', linewidth=2, color='tab:blue')
plt.title("Drift Index (Jensen–Shannon Divergence vs 7-day average)")
plt.xlabel("Date")
plt.ylabel("JSD (0–1)")
plt.ylim(0, 1)
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig("results/drift_index.png")
plt.close()
