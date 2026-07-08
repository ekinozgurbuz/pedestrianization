import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("TkAgg")   # if this fails on your machine, switch to "Agg"
import matplotlib.pyplot as plt

from shapely.geometry import Point, LineString
from shapely import wkt
from matplotlib.lines import Line2D

# ============================================================
# FILE PATHS: change these
# ============================================================
nodes_file = r"C:\Users\Ekin\Desktop\ChapterThreeNew\London Large_nodes_model.xlsx"
edges_file = r"C:\Users\Ekin\Desktop\ChapterThreeNew\London Large_edges_model.xlsx"

# If your Excel files use sheet names, set them here
nodes_sheet = 0   # or "nodes"
edges_sheet = 0   # or "edges"

# ============================================================
# SETTINGS
# ============================================================
show_edge_ids = False   # set False if too cluttered
label_only_long_edges = False
min_length_for_label = 40

# Toggle this to enable/disable extension plan colouring
show_extension_plan = False

#If you want to highlight some specific edges on the plot
# Edge IDs to colour as extension plan
extension_edge_ids = {10, 48, 57, 91, 90, 19, 20, 83, 476, 472, 516, 500, 502, 344, 345, 85, 539}

# ============================================================
# READ EXCELS
# ============================================================
nodes_df = pd.read_excel(nodes_file, sheet_name=nodes_sheet)
edges_df = pd.read_excel(edges_file, sheet_name=edges_sheet)

print("Loaded nodes:", nodes_df.shape)
print("Loaded edges:", edges_df.shape)
print("Node columns:", list(nodes_df.columns))
print("Edge columns:", list(edges_df.columns))

# ============================================================
# BASIC CHECKS
# ============================================================
required_node_cols = {"node_id", "x", "y"}
required_edge_cols = {"edge_id", "start_node_id", "end_node_id"}

missing_node = required_node_cols - set(nodes_df.columns)
missing_edge = required_edge_cols - set(edges_df.columns)

if missing_node:
    raise ValueError(f"Missing node columns: {missing_node}")
if missing_edge:
    raise ValueError(f"Missing edge columns: {missing_edge}")

# ============================================================
# BUILD NODE GEOMETRY
# ============================================================
nodes_df = nodes_df.copy()
nodes_df["geometry"] = nodes_df.apply(lambda r: Point(r["x"], r["y"]), axis=1)
nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry="geometry")

# ============================================================
# BUILD EDGE GEOMETRY
# If geometry_wkt exists, use it.
# Otherwise, create straight line from start/end node coordinates.
# ============================================================
edges_df = edges_df.copy()

if "geometry_wkt" in edges_df.columns:
    edges_df["geometry"] = edges_df["geometry_wkt"].apply(wkt.loads)
else:
    node_xy = nodes_df.set_index("node_id")[["x", "y"]].to_dict("index")

    def make_linestring(row):
        s = int(row["start_node_id"])
        t = int(row["end_node_id"])

        if s not in node_xy:
            raise ValueError(f"start_node_id {s} not found in nodes file")
        if t not in node_xy:
            raise ValueError(f"end_node_id {t} not found in nodes file")

        p1 = node_xy[s]
        p2 = node_xy[t]
        return LineString([(p1["x"], p1["y"]), (p2["x"], p2["y"])])

    edges_df["geometry"] = edges_df.apply(make_linestring, axis=1)

edges_gdf = gpd.GeoDataFrame(edges_df, geometry="geometry")

# ============================================================
# SANITY CHECKS
# ============================================================
node_id_set = set(nodes_gdf["node_id"])
bad_start = (~edges_gdf["start_node_id"].isin(node_id_set)).sum()
bad_end = (~edges_gdf["end_node_id"].isin(node_id_set)).sum()

print("Bad start node refs:", int(bad_start))
print("Bad end node refs:", int(bad_end))
print("Null edge geometries:", int(edges_gdf.geometry.isna().sum()))
print("Null node geometries:", int(nodes_gdf.geometry.isna().sum()))

# ============================================================
# PREPARE EDGE FLAGS
# ============================================================
if "currently_ped" not in edges_gdf.columns:
    edges_gdf["currently_ped"] = False

edges_gdf["currently_ped"] = (
    edges_gdf["currently_ped"]
    .fillna(False)
    .replace({1: True, 0: False, "1": True, "0": False, "True": True, "False": False})
    .astype(bool)
)

edges_gdf["edge_id"] = pd.to_numeric(edges_gdf["edge_id"], errors="coerce")

if show_extension_plan:
    edges_gdf["is_extension_plan"] = edges_gdf["edge_id"].isin(extension_edge_ids)
else:
    edges_gdf["is_extension_plan"] = False

# ============================================================
# SPLIT EDGES FOR PLOTTING
# ============================================================
edges_extension = edges_gdf[edges_gdf["is_extension_plan"]].copy()
edges_regular = edges_gdf[~edges_gdf["is_extension_plan"]].copy()

edges_regular_nonped = edges_regular[~edges_regular["currently_ped"]].copy()
edges_regular_ped = edges_regular[edges_regular["currently_ped"]].copy()

# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(12, 12))

# Non-pedestrian edges: solid black
if len(edges_regular_nonped) > 0:
    edges_regular_nonped.plot(ax=ax, linewidth=1.0, color="black", linestyle="-")

# Already pedestrianised edges: dashed indianred
if len(edges_regular_ped) > 0:
    edges_regular_ped.plot(ax=ax, linewidth=1.2, color="indianred", linestyle="--")

# Extension plan edges: darkviolet
if len(edges_extension) > 0:
    edges_extension.plot(ax=ax, linewidth=1.6, color="darkviolet", linestyle="-")

# Nodes
nodes_gdf.plot(ax=ax, markersize=8, color="black")

# Edge ID labels
if show_edge_ids:
    for row in edges_gdf.itertuples(index=False):
        try:
            length_val = row.length_m if "length_m" in edges_gdf.columns else None
            if label_only_long_edges and length_val is not None and length_val < min_length_for_label:
                continue

            mid = row.geometry.interpolate(0.5, normalized=True)
            ax.text(
                mid.x,
                mid.y,
                str(int(row.edge_id)) if pd.notna(row.edge_id) else "",
                fontsize=6,
                color="red",
                ha="center",
                va="center"
            )
        except Exception:
            pass

ax.set_aspect("equal")
ax.set_title("Existing Graph")
ax.axis("off")

legend_elements = [
    Line2D([0], [0], color="black", lw=1.0, linestyle="-", label="Not pedestrianised"),
    Line2D([0], [0], color="indianred", lw=1.2, linestyle="--", label="Pedestrianised"),
]

if show_extension_plan:
    legend_elements.append(
        Line2D([0], [0], color="darkviolet", lw=1.6, linestyle="-", label="Proposed extension")
    )

ax.legend(
    handles=legend_elements,
    loc="lower left",
    fontsize=15,
    frameon=True,
    borderpad=0.6,
    labelspacing=0.4,
    handlelength=2.0,
    handletextpad=0.6
)

from matplotlib.patches import Rectangle

# Add thin black border around entire figure
fig.add_artist(
    Rectangle(
        (0, 0), 1, 1,                 # full figure (normalized coords)
        transform=fig.transFigure,
        fill=False,
        edgecolor="black",
        linewidth=0.8                # try 0.6–1.2 depending on taste
    )
)

plt.tight_layout()
plt.show()