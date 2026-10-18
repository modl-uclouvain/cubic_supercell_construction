import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "dejavusans"


# ============================================================
# Function to read one result file
# ============================================================
def read_data(file_path):

    rows = []

    with open(file_path, "r") as fp:
        raw_lines = fp.readlines()

    i = 0
    for line in raw_lines:
        if not line.startswith("supercell size"):
            i += 1
            continue
        else:
            i += 2
            break

    for j in range(i, len(raw_lines)):
        line = raw_lines[j]

        try:
            parts = [float(p) for p in line.strip().split(",")]
        except ValueError:
            continue

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

    return df_small


# ============================================================
# Files and panel labels
# ============================================================
structures = {
    "Simple cubic": "./results/SC.txt",
    "Body centered cubic": "./results/BCC.txt",
    "Face centered cubic": "./results/FCC.txt",
    "Hexagonal close packed": "./results/HCP.txt",
}


# ============================================================
# Shared color scale
# ============================================================
norm = Normalize(vmin=0, vmax=0.004)
cmap = "viridis"


# ============================================================
# Create figure
# ============================================================
fig, axes = plt.subplots(
    2, 2,
    figsize=(10, 8),
    sharex=True,
    sharey=True,
)

axes = axes.ravel()


# ============================================================
# Plot each structure
# ============================================================
for ax, (title, file_path) in zip(axes, structures.items()):

    df_small = read_data(file_path)

    print(
        f"{title}: "
        f"number of points = {len(df_small)}, "
        f"max delta_eta = {df_small['delta_eta'].max():.6f}"
    )

    sc = ax.scatter(
        df_small["relative_length_difference"],
        df_small["relative_angle_difference"],
        c=df_small["delta_eta"],
        norm=norm,
        cmap=cmap,
        s=40,
    )

    ax.set_title(title, fontsize=14)

    ax.set_xlim(-0.0025, 0.0525)
    ax.set_ylim(-0.0025, 0.0525)

    ax.tick_params(
        axis="both",
        labelsize=11
    )


# ============================================================
# Shared x/y labels
# ============================================================
fig.supxlabel(
    r"$\Delta L\,/\,V^{1/3}$",
    fontsize=16,
)

fig.supylabel(
    r"$\Delta\theta\,/\,90^\circ$",
    fontsize=16,
)


# ============================================================
# Adjust subplot spacing
# ============================================================
fig.subplots_adjust(
    left=0.10,
    right=0.84,      # right edge of the 2x2 subplot area
    bottom=0.10,
    top=0.94,
    wspace=0.10,
    hspace=0.18,
)


# ============================================================
# Shared colorbar
# ============================================================

# [left, bottom, width, height]
cax = fig.add_axes([
    0.87,   # horizontal position
    0.18,   # vertical position
    0.022,  # width
    0.68,   # height
])

cbar = fig.colorbar(
    sc,
    cax=cax,
)

cbar.set_label(
    r"$\Delta\eta(N)$",
    fontsize=16,
)

cbar.ax.tick_params(labelsize=11)


# ============================================================
# Save
# ============================================================
plt.savefig(
    "SC_BCC_FCC_HCP_scatter_colormap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    "SC_BCC_FCC_HCP_scatter_colormap.pdf",
    bbox_inches="tight",
)

plt.show()