import pandas as pd
import matplotlib.pyplot as plt

rows = []

file_name = 'SC.txt'
file_path = f'../results/{file_name}'
with open(file_path, 'r+') as fp:
    raw_lines = fp.readlines()

i = 0
for line in raw_lines:
    if not line.startswith('supercell size'):
        i += 1
        continue
    else:
        i += 2
        break

for j in range(i, len(raw_lines)):
    line = raw_lines[j]
    parts = [float(p) for p in line.strip().split(',')]
    if len(parts) < 11:
        continue
    rows.append({
        "supercell size": parts[0],
        "delta_eta": parts[1],
        "total P matrix": parts[2],
        "a": parts[3],
        "b": parts[4],
        "c": parts[5],
        "alpha": parts[6],
        "beta": parts[7],
        "gamma": parts[8],
        "relative_length_difference": parts[9],
        "relative_angle_difference": parts[10],
    })

df = pd.DataFrame(rows)

df_small = df[
    (df["relative_length_difference"] < 0.05) &
    (df["relative_angle_difference"] < 0.05)
]

max_delta_eta = df_small["delta_eta"].max()
print(max_delta_eta)

plt.figure(figsize=(6.5, 5.2))

tricf = plt.tricontourf(
    df_small["relative_length_difference"],
    df_small["relative_angle_difference"],
    df_small["delta_eta"],
    levels=18
)

cont = plt.tricontour(
    df_small["relative_length_difference"],
    df_small["relative_angle_difference"],
    df_small["delta_eta"],
    levels=8
)

plt.clabel(cont, inline=True, fontsize=8)

plt.scatter(
    df_small["relative_length_difference"],
    df_small["relative_angle_difference"],
    s=8
)

plt.xlabel(r"$\Delta_L$")
plt.ylabel(r"$\Delta_\theta$")
plt.title(r"Candidate 3: filled contour + contour lines + data points")
plt.colorbar(tricf, label=r"$\Delta\eta$")
plt.tight_layout()

plt.savefig("candidate3_tricontourf_scatter.png", dpi=300, bbox_inches="tight")
plt.show()
