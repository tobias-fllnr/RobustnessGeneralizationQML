import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
num_r_values = 1000
num_iterations = 12
x0 = 0.5

# Generate r values
r_values = np.linspace(3.5, 4, num_r_values)

# Initialize array: shape (timesteps, r_values)
x = np.zeros((num_iterations, num_r_values))
x[0, :] = x0

# Iterate logistic map
for t in range(1, num_iterations):
    x[t, :] = r_values * x[t - 1, :] * (1 - x[t - 1, :])

# Convert to DataFrame, columns labeled by r
df = pd.DataFrame(x, columns=[f"{r:.5f}" for r in r_values])
df.to_csv("./TimeseriesData/logistic_map_chaos.csv", index=False)


#Plot bifurcation diagram

# Parameters
num_r_values = 500
num_iterations = 50
x0 = 0.5

# Generate r values
r_values = np.linspace(0, 4, num_r_values)

# Initialize array: shape (timesteps, r_values)
x = np.zeros((num_iterations, num_r_values))
x[0, :] = x0

# Iterate logistic map
for t in range(1, num_iterations):
    x[t, :] = r_values * x[t - 1, :] * (1 - x[t - 1, :])

# Convert to DataFrame, columns labeled by r
df = pd.DataFrame(x, columns=[f"{r:.5f}" for r in r_values])

# Load the data (make sure it's in the same directory or provide full path)
#df = pd.read_csv("logistic_map.csv")

# Convert column names back to floats (they are strings in CSV)
r_values = [float(col) for col in df.columns]

# Discard transient (e.g., first 50 iterations)
#df_tail = df.tail(50)
df_tail = df.head(50)

# Prepare bifurcation plot
plt.figure(figsize=(5, 3))

# For each r, plot the last 50 x values
for r, col in zip(r_values, df_tail.columns):
    plt.plot([r]*len(df_tail), df_tail[col], ',k', alpha=0.4)  # small black dots

plt.title("Bifurcation Diagram of the Logistic Map")
plt.xlabel("$r$")
plt.ylabel("$x_t$")
plt.tight_layout()
plt.savefig("./Plots/bifurcation_diagram.pdf")