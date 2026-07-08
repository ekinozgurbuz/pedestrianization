from shapely.geometry import Polygon, Point
from shapely.ops import linemerge, unary_union
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import box
ox.settings.use_cache = True
ox.settings.log_console = True

# # -------------------------------------------------
# # Manual Liverpool city-centre bounding box
# # -------------------------------------------------
# north = 53.40702
# south = 53.39999
# west  = -2.99013
# east  = -2.97601
# # -------------------------------------------------
# # Manual London Oxford Street bounding box
# # -------------------------------------------------
# temp_var = "London Large"
# north = 51.51652 + 0.002
# south = 51.51075 - 0.002
# west  = -0.16110 - 0.002
# east  = -0.13937 + 0.002
# # -------------------------------------------------
# # Manual New York bounding box
# # -------------------------------------------------
# temp_var = "New York Large"
# north = 40.760829 + 0.002 *2
# south = 40.755059 - 0.002 *2
# west  = -73.996448 - 0.002 *2
# east  = -73.974718 + 0.002 *2

# -------------------------------------------------
# Manual Paris bounding box
# -------------------------------------------------
temp_var = "Paris Large"
north = 48.86912 +0.001
south = 48.86387 -0.001
west  = 2.33794 -0.001
east  = 2.36094 +0.001


# ============================================================
# USER-ADJUSTABLE SETTINGS
# ============================================================

#To exclude edges shorter than specified length (m)
MIN_EDGE_LENGTH_M = 10.0

# Weights for calculating the attractivity/footfall across all attractiveness components
ALPHA_COMPONENTS = {
    "retail": 1.0,
    "shops": 1.0,
    "tourism": 1.0,
    "attractions": 1.0,
    "hotels": 1.0,
    "transit_access": 1.0,
    "food_leisure": 1.0,
    "centrality": 1.0,
}

# Optional global scale for alpha/profit
ALPHA_SCALE = 100.0

# ============================================================
# OSM NETWORK FILTER
# Excludes service roads from the extracted network.
# If later needed, add service back explicitly.
# ============================================================

#Which type of roads to include
custom_filter = """
["highway"~"primary|secondary|tertiary|unclassified|residential|living_street|pedestrian"]
["area"!~"yes"]
"""

# ============================================================
# FEATURE TAGS FOR SIMPLE ATTRACTIVENESS CONSTRUCTION
# ============================================================

feature_tags = {
    "shop": True,
    "tourism": True,
    "amenity": True,
    "leisure": True,
    "railway": True,
    "public_transport": True,
    "highway": True,
    "building": True,
    "landuse": True,
    "historic": True,
}

# ============================================================
# HELPERS
# ============================================================

def first_value(x):
    if isinstance(x, list):
        return x[0] if len(x) > 0 else None
    return x

def first_display_name(x):
    if isinstance(x, (list, tuple)):
        for item in x:
            if item is None:
                continue
            s = str(item).strip()
            if s != "":
                return s
        return None
    if x is None:
        return None
    s = str(x).strip()
    return s if s != "" else None

def minmax(s):
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)

def subset_features(df, tag_col, vals=None):
    if tag_col not in df.columns:
        return df.iloc[0:0].copy()
    s = df[tag_col]
    if vals is None:
        return df[s.notnull()].copy()
    return df[s.isin(vals)].copy()

def assign_features_to_edges_by_intersection_else_nearest(edges_gdf, feature_gdf, out_name):
    """
    Assign each feature to edges as follows:
      1) if it intersects one or more edges, count it on intersecting edges
      2) otherwise assign it to the nearest edge

    Works for points, lines, and polygons.
    """
    edge_keys = edges_gdf[["u", "v", "key", "geometry"]].copy()

    if len(feature_gdf) == 0:
        result = edge_keys[["u", "v", "key"]].copy()
        result[out_name] = 0
        return result

    feat = feature_gdf[["geometry"]].copy().reset_index(drop=True)
    feat["feature_id"] = np.arange(len(feat))

    # Step 1: assign by intersection
    inter = gpd.sjoin(
        feat[["feature_id", "geometry"]],
        edge_keys,
        how="inner",
        predicate="intersects"
    )

    inter_counts = (
        inter.groupby(["u", "v", "key"])
        .size()
        .reset_index(name=out_name)
    )

    assigned_ids = set(inter["feature_id"].unique()) if len(inter) > 0 else set()

    # Step 2: assign remaining features to nearest edge
    rem = feat.loc[~feat["feature_id"].isin(assigned_ids), ["feature_id", "geometry"]].copy()

    if len(rem) > 0:
        nearest = gpd.sjoin_nearest(
            rem,
            edge_keys,
            how="left",
            distance_col="_dist_to_edge"
        )

        nearest_counts = (
            nearest.groupby(["u", "v", "key"])
            .size()
            .reset_index(name=out_name)
        )
    else:
        nearest_counts = edge_keys[["u", "v", "key"]].copy()
        nearest_counts[out_name] = 0
        nearest_counts = nearest_counts.iloc[0:0].copy()

    # Combine
    if len(inter_counts) == 0 and len(nearest_counts) == 0:
        result = edge_keys[["u", "v", "key"]].copy()
        result[out_name] = 0
        return result

    combined = pd.concat([inter_counts, nearest_counts], ignore_index=True)
    combined = (
        combined.groupby(["u", "v", "key"], as_index=False)[out_name]
        .sum()
    )

    return combined

def build_edge_centrality(edges_df):
    """
    Compute Freeman edge betweenness centrality on the cleaned undirected graph.
    OSM does not provide centrality as a tag; it is computed from the graph.
    """
    G_central = nx.Graph()

    for row in edges_df.itertuples(index=False):
        u = int(row.start_node_id)
        v = int(row.end_node_id)
        length_val = float(row.length_m) if pd.notna(row.length_m) else 1.0

        # If parallel edges exist, keep the shortest for path-based centrality
        if G_central.has_edge(u, v):
            if length_val < G_central[u][v]["length"]:
                G_central[u][v]["length"] = length_val
        else:
            G_central.add_edge(u, v, length=length_val)

    edge_bc = nx.edge_betweenness_centrality(
        G_central,
        normalized=True,
        weight="length"
    )

    def lookup_bc(row):
        uv = tuple(sorted((int(row["start_node_id"]), int(row["end_node_id"]))))
        return edge_bc.get(uv, 0.0)

    return edges_df.apply(lookup_bc, axis=1)

# ============================================================
# 1) DOWNLOAD NETWORK
# ============================================================

G = ox.graph_from_bbox(
    bbox=(west, south, east, north),
    custom_filter=custom_filter,
    simplify=False,
)

# Unprojected graph and GeoDataFrames
nodes_gdf_ll, edges_gdf_ll = ox.graph_to_gdfs(G, nodes=True, edges=True)

# Project graph to metric CRS
G_proj = ox.project_graph(G)

# Simplify graph
G_model = ox.simplification.simplify_graph(G_proj)
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_model, nodes=True, edges=True)

print("Unprojected CRS:", edges_gdf_ll.crs)
print("Projected CRS:", edges_gdf.crs)
print("Nodes / edges (simplified):", nodes_gdf.shape, edges_gdf.shape)

# ============================================================
# 2) CLEAN EDGE TABLE
# ============================================================

edges = edges_gdf.reset_index().copy()
nodes = nodes_gdf.reset_index().copy()

keep_cols = [
    "u", "v", "key", "osmid", "name", "highway", "oneway",
    "length", "geometry", "access", "motor_vehicle", "foot"
]
keep_cols = [c for c in keep_cols if c in edges.columns]
edges = edges[keep_cols].copy()

if "length" in edges.columns:
    edges = edges.rename(columns={"length": "length_m"})
else:
    edges["length_m"] = edges.geometry.length

# Use projected geometry length as definitive metric length
edges["length_m"] = edges.geometry.length

edges["highway_main"] = edges["highway"].apply(first_value)
edges["edge_name"] = edges["name"].apply(first_display_name) if "name" in edges.columns else None

# Current pedestrian status
ped_highway_vals = {"pedestrian", "living_street"}
edges["currently_ped"] = edges["highway_main"].isin(ped_highway_vals)

if "motor_vehicle" in edges.columns:
    edges.loc[edges["motor_vehicle"].isin(["no", "private"]), "currently_ped"] = True

# Optional: only upgrade currently_ped using foot if road is already pedestrian-oriented
if "foot" in edges.columns:
    edges.loc[
        edges["highway_main"].isin({"pedestrian", "living_street"}) &
        edges["foot"].isin(["yes", "designated"]),
        "currently_ped"
    ] = True

# Keep meaningful classes already controlled by custom_filter, but check again
allowed_highways = {
    "primary", "secondary", "tertiary",
    "unclassified", "residential",
    "living_street", "pedestrian"
}
edges = edges[edges["highway_main"].isin(allowed_highways)].copy()

# Remove tiny segments
edges = edges[edges["length_m"] >= MIN_EDGE_LENGTH_M].copy()

print("Remaining edges after road-class / length filtering:", len(edges))
print(edges["highway_main"].value_counts())

# ============================================================
# 3) BUILD NODE MAPPING
# ============================================================

nodes_simple = nodes[["osmid", "x", "y", "geometry"]].copy()
nodes_simple = nodes_simple.rename(columns={"osmid": "osm_node_id"}).drop_duplicates("osm_node_id")
nodes_simple = nodes_simple.reset_index(drop=True)
nodes_simple["node_id"] = np.arange(1, len(nodes_simple) + 1)

osm_to_new = dict(zip(nodes_simple["osm_node_id"], nodes_simple["node_id"]))

edges_model = edges.copy()
edges_model["start_node_id"] = edges_model["u"].map(osm_to_new)
edges_model["end_node_id"] = edges_model["v"].map(osm_to_new)

edges_model = edges_model[
    edges_model["start_node_id"].notnull() &
    edges_model["end_node_id"].notnull()
].copy()

edges_model["start_node_id"] = edges_model["start_node_id"].astype(int)
edges_model["end_node_id"] = edges_model["end_node_id"].astype(int)

# Remove self-loops
edges_model = edges_model[
    edges_model["start_node_id"] != edges_model["end_node_id"]
].copy()

# Keep only largest connected component
G_clean = nx.Graph()
for row in edges_model.itertuples(index=False):
    G_clean.add_edge(
        int(row.start_node_id),
        int(row.end_node_id),
        length=float(row.length_m)
    )

if G_clean.number_of_edges() == 0:
    raise ValueError("No usable edges remain after cleaning.")

largest_cc = max(nx.connected_components(G_clean), key=len)

edges_model = edges_model[
    edges_model["start_node_id"].isin(largest_cc) &
    edges_model["end_node_id"].isin(largest_cc)
].copy()

nodes_simple = nodes_simple[nodes_simple["node_id"].isin(largest_cc)].copy()

edges_model = edges_model.reset_index(drop=True)
edges_model["edge_id"] = np.arange(1, len(edges_model) + 1)

# If edge name is missing, fill safely
edges_model["edge_name"] = edges_model["edge_name"].where(
    edges_model["edge_name"].notnull(),
    "Unnamed street"
)

print("Nodes after connectivity cleanup:", len(nodes_simple))
print("Edges after connectivity cleanup:", len(edges_model))

# ============================================================
# 4) STUDY AREA POLYGON FOR OSM FEATURE QUERY
# Use exact manual bbox, not union envelope
# features_from_polygon expects EPSG:4326
# ============================================================

study_area = box(west, south, east, north)

features = ox.features_from_polygon(study_area, tags=feature_tags).copy()
features = features.to_crs(edges_gdf.crs)

features = features.reset_index()
features = features[features.geometry.notnull()].copy()
features = gpd.GeoDataFrame(features, geometry="geometry", crs=edges_gdf.crs)

print("OSM feature rows:", len(features))
print("Feature columns:", features.columns.tolist())

# Helper tag columns
features["shop_tag"] = features.get("shop")
features["tourism_tag"] = features.get("tourism")
features["amenity_tag"] = features.get("amenity")
features["leisure_tag"] = features.get("leisure")
features["railway_tag"] = features.get("railway")
features["pt_tag"] = features.get("public_transport")
features["highway_tag"] = features.get("highway")
features["building_tag"] = features.get("building")
features["landuse_tag"] = features.get("landuse")
features["historic_tag"] = features.get("historic")

edges_model = gpd.GeoDataFrame(edges_model, geometry="geometry", crs=edges_gdf.crs)

# ============================================================
# 5) DEFINE SIMPLE ATTRACTIVENESS COMPONENTS
# Equal weights across components
# ============================================================

# Retail proxy
retail_subset = pd.concat([
    subset_features(features, "building_tag", {"retail"}),
    subset_features(features, "landuse_tag", {"retail"}),
    subset_features(features, "amenity_tag", {"marketplace"}),
], ignore_index=True)
retail_subset = gpd.GeoDataFrame(retail_subset, geometry="geometry", crs=features.crs)

# Shops proxy
shops_subset = subset_features(features, "shop_tag", None)

# Tourism proxy
tourism_subset = subset_features(features, "tourism_tag", None)

# Attractions proxy
attractions_subset = pd.concat([
    subset_features(features, "tourism_tag", {"attraction", "museum", "gallery", "viewpoint", "zoo", "theme_park"}),
    subset_features(features, "historic_tag", None),
], ignore_index=True)
attractions_subset = gpd.GeoDataFrame(attractions_subset, geometry="geometry", crs=features.crs)

# Hotels proxy
hotels_subset = pd.concat([
    subset_features(features, "tourism_tag", {"hotel", "guest_house", "hostel", "motel"}),
    subset_features(features, "building_tag", {"hotel"}),
], ignore_index=True)
hotels_subset = gpd.GeoDataFrame(hotels_subset, geometry="geometry", crs=features.crs)

# Transit access proxy
transit_subset = pd.concat([
    subset_features(features, "railway_tag", None),
    subset_features(features, "pt_tag", None),
    subset_features(features, "highway_tag", {"bus_stop"}),
], ignore_index=True)
transit_subset = gpd.GeoDataFrame(transit_subset, geometry="geometry", crs=features.crs)

# Food and leisure proxy
food_leisure_subset = pd.concat([
    subset_features(features, "amenity_tag", {"restaurant", "cafe", "bar", "pub", "fast_food"}),
    subset_features(features, "leisure_tag", None),
], ignore_index=True)
food_leisure_subset = gpd.GeoDataFrame(food_leisure_subset, geometry="geometry", crs=features.crs)

# ============================================================
# 6) ASSIGN COMPONENT COUNTS TO EDGES
# ============================================================

component_subsets = {
    "retail": retail_subset,
    "shops": shops_subset,
    "tourism": tourism_subset,
    "attractions": attractions_subset,
    "hotels": hotels_subset,
    "transit_access": transit_subset,
    "food_leisure": food_leisure_subset,
}

feature_count_tables = []
availability_rows = []

for comp_name, comp_subset in component_subsets.items():
    availability_rows.append({
        "component": comp_name,
        "n_osm_objects": len(comp_subset),
    })

    df_cnt = assign_features_to_edges_by_intersection_else_nearest(
        edges_model,
        comp_subset,
        f"{comp_name}_cnt"
    )
    feature_count_tables.append(df_cnt)

availability_df = pd.DataFrame(availability_rows).sort_values(
    by=["n_osm_objects", "component"],
    ascending=[False, True]
)

print("--------------------------------------------------")
print("SANITY CHECK: simple attractiveness components")
print(availability_df.to_string(index=False))

missing_components = availability_df.loc[
    availability_df["n_osm_objects"] == 0, "component"
].tolist()

if missing_components:
    print("WARNING: the following components are absent in the current study area:")
    print(missing_components)

for df_cnt in feature_count_tables:
    edges_model = edges_model.merge(df_cnt, on=["u", "v", "key"], how="left")

count_cols = [f"{comp_name}_cnt" for comp_name in component_subsets.keys()]
edges_model[count_cols] = edges_model[count_cols].fillna(0)

# ============================================================
# 7) COMPUTE EDGE CENTRALITY
# OSM has no centrality tag.
# We compute Freeman edge betweenness centrality from the graph.
# ============================================================

edges_model["centrality_raw"] = build_edge_centrality(edges_model)

# ============================================================
# 8) NORMALISE COMPONENTS AND COMPUTE ALPHA
# Equal-weight average, baseline = 0
# ============================================================

for comp_name in component_subsets.keys():
    edges_model[f"{comp_name}_n"] = minmax(edges_model[f"{comp_name}_cnt"])

edges_model["centrality_n"] = minmax(edges_model["centrality_raw"])

weight_sum = sum(ALPHA_COMPONENTS.values())
component_weights = {
    k: v / weight_sum for k, v in ALPHA_COMPONENTS.items()
}

edges_model["alpha_raw"] = 0.0
for comp_name, w in component_weights.items():
    source_col = f"{comp_name}_n"
    edges_model["alpha_raw"] += w * edges_model[source_col]

# baseline = 0
edges_model["alpha"] = ALPHA_SCALE * edges_model["alpha_raw"]
edges_model["profit"] = edges_model["alpha"]

print("--------------------------------------------------")
print("Component weights (uniform after normalisation):")
for k, v in component_weights.items():
    print(f"{k:20s} {v:.4f}")

print(edges_model[[
    "edge_id", "edge_name", "length_m",
    "alpha_raw", "alpha", "profit", "centrality_raw"
]].head())

# ============================================================
# 9) ADD SIMPLE MODEL FIELDS
# ============================================================

edges_model["demand"] = 100.0
edges_model["t_motorised"] = 1.0
edges_model["t_pedestrianised"] = 1.0

# ============================================================
# 10) OPTIONAL DEDUPLICATION OF UNDIRECTED PARALLEL EDGES
# Keep the one with highest alpha
# ============================================================

edges_model["uv_key"] = edges_model.apply(
    lambda r: tuple(sorted((int(r["start_node_id"]), int(r["end_node_id"])))),
    axis=1
)

before_dedup = len(edges_model)

edges_model = edges_model.sort_values(
    by=["uv_key", "alpha"],
    ascending=[True, False]
)

edges_model = edges_model.drop_duplicates(
    subset=["uv_key"],
    keep="first"
).copy()

edges_model = edges_model.drop(columns=["uv_key"], errors="ignore")
edges_model = edges_model.reset_index(drop=True)
edges_model["edge_id"] = np.arange(1, len(edges_model) + 1)

print("--------------------------------------------------")
print("Duplicate undirected edge removal")
print("Before:", before_dedup)
print("After :", len(edges_model))
print("Removed:", before_dedup - len(edges_model))

# Keep only nodes actually referenced now
used_nodes = set(edges_model["start_node_id"]).union(set(edges_model["end_node_id"]))
nodes_simple = nodes_simple[nodes_simple["node_id"].isin(used_nodes)].copy()

# ============================================================
# 11) PREPARE EXPORT TABLES
# ============================================================

edges_export = edges_model.copy()
nodes_export = nodes_simple.copy()

edges_export["geometry_wkt"] = edges_export.geometry.to_wkt()
nodes_export["geometry_wkt"] = nodes_export.geometry.to_wkt()

edge_cols = [
    "edge_id",
    "edge_name",
    "start_node_id",
    "end_node_id",
    "length_m",
    "highway_main",
    "currently_ped",
    "retail_cnt",
    "shops_cnt",
    "tourism_cnt",
    "attractions_cnt",
    "hotels_cnt",
    "transit_access_cnt",
    "food_leisure_cnt",
    "centrality_raw",
    "alpha_raw",
    "alpha",
    "profit",
    "demand",
    "t_motorised",
    "t_pedestrianised",
    "geometry_wkt",
]
edge_cols = [c for c in edge_cols if c in edges_export.columns]
edges_out = edges_export[edge_cols].copy()

node_cols = ["node_id", "x", "y", "geometry_wkt"]
node_cols = [c for c in node_cols if c in nodes_export.columns]
nodes_out = nodes_export[node_cols].copy()

# Round numeric columns for export
float_cols_edges = edges_out.select_dtypes(include=["float64", "float32"]).columns.tolist()
edges_out[float_cols_edges] = edges_out[float_cols_edges].round(4)

float_cols_nodes = nodes_out.select_dtypes(include=["float64", "float32"]).columns.tolist()
nodes_out[float_cols_nodes] = nodes_out[float_cols_nodes].round(2)

print("--------------------------------------------------")
print("Final export check")
print("Nodes:", len(nodes_out))
print("Edges:", len(edges_out))
print(edges_out.head())

edges_out.to_excel(f"{temp_var}_edges_model.xlsx", index=False)
nodes_out.to_excel(f"{temp_var}_nodes_model.xlsx", index=False)

print(f"Saved {temp_var}_edges_model.xlsx and {temp_var}_nodes_model.xlsx")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import geopandas as gpd

# Ensure GeoDataFrame
edges_plot = gpd.GeoDataFrame(edges_model, geometry="geometry", crs=edges_model.crs)
nodes_plot = gpd.GeoDataFrame(nodes_simple, geometry="geometry", crs=nodes_simple.crs)

fig, ax = plt.subplots(figsize=(10, 10))

# Plot edges
edges_plot.plot(ax=ax, linewidth=0.8, color="black")

# Plot nodes
nodes_plot.plot(ax=ax, markersize=8, color="blue")

# -------------------------------------------------
# Add edge_id labels
# -------------------------------------------------
for row in edges_plot.itertuples(index=False):
    if row.geometry is None or row.geometry.is_empty:
        continue

    try:
        mid = row.geometry.interpolate(0.5, normalized=True)
    except Exception:
        mid = row.geometry.centroid

    ax.text(
        mid.x,
        mid.y,
        str(row.edge_id),
        fontsize=6,
        color="red",
        ha="center",
        va="center"
    )

ax.set_aspect("equal")
ax.set_title(f"{temp_var} cleaned graph (edge IDs)")
ax.axis("off")

plt.tight_layout()
plt.show()