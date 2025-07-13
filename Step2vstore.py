import os
import json
import hashlib
import pickle
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document
from langchain.text_splitter import NLTKTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.Interface import Interface_Static
from OCC.Core.TDF import TDF_Label
from OCC.Core.TDataStd import TDataStd_Name
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from tqdm import tqdm
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common

# Load configuration
with open("config.json", "r") as config_file:
    config = json.load(config_file)

step_directory = config["data_sources"].get("step", {}).get("directory", "./STEP_FILES")
vector_store_path = config["vector_store"]["path"]

os.makedirs(vector_store_path, exist_ok=True)

def create_vstore():
    return Qdrant.from_documents(
        [Document(page_content='')],
        HuggingFaceEmbeddings(),
        path=vector_store_path,
        collection_name='all',
    )

vstore = create_vstore()

HASH_STORE_PATH = os.path.join(os.path.dirname(__file__), "step_hashes.pkl")

def compute_hash(content):
    if isinstance(content, str):
        return hashlib.sha512(content.encode('utf-8')).hexdigest()
    return None

def compute_file_hash(file_path):
    hasher = hashlib.sha512()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_hash_set():
    if os.path.exists(HASH_STORE_PATH):
        with open(HASH_STORE_PATH, "rb") as f:
            return pickle.load(f)
    return set()

def save_hash_set(hash_set):
    with open(HASH_STORE_PATH, "wb") as f:
        pickle.dump(hash_set, f)

def get_bounding_box(solid):
    box = Bnd_Box()
    # Use the static method to avoid deprecation warning
    from OCC.Core import BRepBndLib
    BRepBndLib.brepbndlib.Add(solid, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (xmin, ymin, zmin), (xmax, ymax, zmax)

def describe_shape(min_point, max_point):
    dims = [abs(max_point[i] - min_point[i]) for i in range(3)]
    if max(dims) - min(dims) < 1.0:
        return f"Likely Cube (dims: {dims})"
    elif dims.count(min(dims)) == 2:
        return f"Likely Cylinder (dims: {dims})"
    else:
        return f"Irregular Shape (dims: {dims})"

def classify_position(cog):
    def axis_label(v):
        if v > 10: return 'Right'
        elif v < -10: return 'Left'
        else: return 'Center'
    x, y, z = cog.X(), cog.Y(), cog.Z()
    return f"{axis_label(x)}-{axis_label(y)}-{axis_label(z)}"

def get_role(spatial_position):
    axes = spatial_position.split('-')
    labels = ['Left', 'Center', 'Right']
    counts = [axes.count(label) for label in labels]
    if counts[1] == 3:
        return 'Core'
    elif counts[1] == 2:
        return 'Face Center'
    elif counts[1] == 1:
        return 'Edge'
    elif counts[1] == 0:
        return 'Corner'
    else:
        return 'Unknown'

position_map = {'Left': -1, 'Center': 0, 'Right': 1}

def label_to_coords(label):
    x, y, z = label.split('-')
    return (position_map[x], position_map[y], position_map[z])

def normalize_units(value, unit):
    conversion = {
        "MM": 1.0,
        "INCH": 25.4,
        "UM": 0.001
    }
    return value * conversion.get(unit.upper(), 1.0)

def get_metadata(reader):
    # Try to extract STEP metadata
    try:
        units_raw = Interface_Static.Static("length.unit")
        units = units_raw.upper() if units_raw else "MM"
    except Exception:
        units = "MM"
    # Placeholder for other metadata extraction (Name, Author, Material, Description)
    # Real extraction would require deeper STEP parsing, which is complex
    return {
        "units": units,
        "author": None,
        "material": None,
        "description": None,
        "name": None
    }

def solids_share_face(s1, s2):
    """Returns True if solids share common geometry."""
    try:
        common = BRepAlgoAPI_Common(s1, s2)
        common.Build()
        return not common.Shape().IsNull()
    except Exception:
        return False

def bounding_boxes_overlap(bb1, bb2, margin=1.0):
    # bb1, bb2: (xmin, ymin, zmin, xmax, ymax, zmax)
    return all(
        bb1[i] <= bb2[i+3] + margin and bb2[i] <= bb1[i+3] + margin
        for i in range(3)
    )

def build_proximity_clusters(bboxes):
    clusters = []
    visited = set()
    for i, bb1 in enumerate(bboxes):
        if i in visited:
            continue
        cluster = [i]
        for j, bb2 in enumerate(bboxes):
            if i != j and bounding_boxes_overlap(bb1, bb2):
                cluster.append(j)
                visited.add(j)
        clusters.append(cluster)
    return clusters

def is_through_hole(face, bbox, axis):
    # Simple Z-direction check for through-hole
    z_min, z_max = bbox[2], bbox[5]
    axis_z = axis.Direction().Z()
    location_z = axis.Location().Z()
    return abs(location_z - z_min) < 1.0 and abs(location_z + axis_z - z_max) < 1.0

def is_edge_hole(axis_origin, bbox, threshold=0.1):
    for i in range(3):
        size = bbox[i+3] - bbox[i]
        margin = size * threshold
        coord = axis_origin.Coord(i+1)
        if abs(coord - bbox[i]) < margin or abs(coord - bbox[i+3]) < margin:
            return True
    return False

try:
    from scipy.spatial import KDTree
except ImportError:
    KDTree = None  # Optional, only used for hole clustering

def cluster_holes(hole_coords, distance_threshold=5.0):
    if not KDTree or not hole_coords:
        return []
    tree = KDTree(hole_coords)
    clusters = []
    visited = set()
    for i in range(len(hole_coords)):
        if i in visited:
            continue
        idxs = tree.query_ball_point(hole_coords[i], r=distance_threshold)
        cluster = [hole_coords[j] for j in idxs]
        clusters.append(cluster)
        visited.update(idxs)
    return clusters

def extract_faces(shape, unit):
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    while face_explorer.More():
        face = face_explorer.Current()
        area = BRep_Tool.Surface(face).Area() if hasattr(BRep_Tool.Surface(face), 'Area') else None
        area_mm2 = normalize_units(area, unit) if area else None
        surface = BRepAdaptor_Surface(face, True)
        surf_type = surface.GetType()
        faces.append({
            "area": area_mm2,
            "type": surf_type,
            "is_cylinder": surf_type == GeomAbs_Cylinder,
            "face": face
        })
        face_explorer.Next()
    return faces

def extract_edges(shape, unit):
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    edges = []
    while edge_explorer.More():
        edge = edge_explorer.Current()
        curve = BRepAdaptor_Curve(edge)
        length = None
        try:
            length = GCPnts_AbscissaPoint.Length(curve)
        except Exception:
            pass  # If curve is invalid or not measurable, skip length
        length_mm = normalize_units(length, unit) if length else None
        edges.append({
            "length": length_mm,
            "type": curve.GetType(),
            "edge": edge
        })
        edge_explorer.Next()
    return edges

def detect_holes(faces, unit, bbox=None):
    holes = []
    hole_coords = []
    for f in faces:
        if f["is_cylinder"]:
            surface = BRepAdaptor_Surface(f["face"], True)
            radius = normalize_units(surface.Cylinder().Radius(), unit)
            axis = surface.Cylinder().Axis()
            axis_origin = axis.Location()
            hole_info = {
                "radius": radius,
                "axis": axis,
                "face": f["face"],
                "axis_origin": axis_origin
            }
            if bbox:
                # Tag through-hole
                if is_through_hole(f["face"], bbox, axis):
                    hole_info["type"] = "Through-hole"
                elif is_edge_hole(axis_origin, bbox):
                    hole_info["type"] = "Edge-hole"
                else:
                    hole_info["type"] = "Blind-hole"
            holes.append(hole_info)
            hole_coords.append((axis_origin.X(), axis_origin.Y(), axis_origin.Z()))
    # Pattern detection
    clusters = cluster_holes(hole_coords)
    for idx, cluster in enumerate(clusters):
        for h in cluster:
            # Mark pattern holes (by proximity)
            for hole in holes:
                if (hole["axis_origin"].X(), hole["axis_origin"].Y(), hole["axis_origin"].Z()) == h:
                    hole["pattern_cluster"] = idx + 1
    return holes

def get_solid_name(solid):
    # Semantic label extraction not available in this mode
    return None

def step_to_text(file_path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path)
    if status != 1:
        raise ValueError(f"STEP file read error: {file_path} (status={status})")

    reader.TransferRoot()
    shape = reader.OneShape()

    # Metadata extraction
    metadata = get_metadata(reader)
    unit = metadata["units"]
    unit_str = unit if unit else "MM"

    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    output = []
    part_name = os.path.basename(file_path)
    output.append(f"Part: {part_name}")
    output.append(f"Units: {unit_str}")
    if metadata["author"]:
        output.append(f"Author: {metadata['author']}")
    if metadata["material"]:
        output.append(f"Material: {metadata['material']}")
    if metadata["description"]:
        output.append(f"Description: {metadata['description']}")
    output.append("All dimensions normalized to millimeters (mm).")

    solids = []
    bboxes = []
    solid_positions = {}
    solid_outputs = []
    solid_roles = {}
    solid_volumes = []
    count = 0
    role_counts = {"Face Center": 0, "Edge": 0, "Corner": 0, "Core": 0, "Unknown": 0}
    pattern_map = {}

    while explorer.More():
        solid = explorer.Current()
        solids.append(solid)
        props = GProp_GProps()
        brepgprop.VolumeProperties(solid, props)
        volume = normalize_units(props.Mass(), unit)
        solid_volumes.append(volume)
        cog = props.CentreOfMass()
        min_pt, max_pt = get_bounding_box(solid)
        min_pt = tuple(normalize_units(x, unit) for x in min_pt)
        max_pt = tuple(normalize_units(x, unit) for x in max_pt)
        bbox = min_pt + max_pt
        bboxes.append(bbox)
        shape_desc = describe_shape(min_pt, max_pt)

        # Semantic label
        solid_name = get_solid_name(solid)
        solid_label = solid_name if solid_name else f"Solid #{count+1}"

        # Feature extraction
        faces = extract_faces(solid, unit)
        edges = extract_edges(solid, unit)
        holes = detect_holes(faces, unit, bbox)
        num_faces = len(faces)
        num_edges = len(edges)
        num_holes = len(holes)

        # Pattern detection (simple: count repeated volumes)
        pattern_key = f"{round(volume,2)}_{round(cog.X(),2)}_{round(cog.Y(),2)}_{round(cog.Z(),2)}"
        pattern_map.setdefault(pattern_key, []).append(solid_label)

        # Role/adjacency logic (config check)
        role_enabled = config.get("step_analysis", {}).get("role_map", False)
        if role_enabled:
            position = classify_position(cog)
            role = get_role(position)
            solid_positions[count + 1] = label_to_coords(position)
            solid_roles[count + 1] = role
            if role in role_counts:
                role_counts[role] += 1
            else:
                role_counts["Unknown"] += 1
        else:
            position = None
            role = None

        solid_output = []
        solid_output.append(f"\n{solid_label}:")
        solid_output.append(f"Volume: {volume:.2f} mm^3")
        solid_output.append(f"Center of Gravity: ({normalize_units(cog.X(), unit):.2f}, {normalize_units(cog.Y(), unit):.2f}, {normalize_units(cog.Z(), unit):.2f})")
        solid_output.append(f"Bounding Box: min {min_pt}, max {max_pt}")
        solid_output.append(f"Shape Description: {shape_desc}")
        solid_output.append(f"Faces: {num_faces}, Edges: {num_edges}, Holes: {num_holes}")
        if num_holes > 0:
            for idx, h in enumerate(holes):
                hole_desc = f"  Hole {idx+1}: radius={h['radius']:.2f} mm"
                if "type" in h:
                    hole_desc += f" — {h['type']}"
                if "pattern_cluster" in h:
                    hole_desc += f" (Pattern Cluster {h['pattern_cluster']})"
                solid_output.append(hole_desc)
        if position:
            solid_output.append(f"Spatial Position: {position}")
        if role:
            solid_output.append(f"Role: {role}")

        # Inferred function heuristic
        inferred_role = "Auxiliary Component"
        if num_holes >= 10 and volume > 10000:
            inferred_role = "Mounting Plate"
        elif role == "Core":
            inferred_role = "Main Housing"
        elif solid_name and "flange" in solid_name.lower():
            inferred_role = "Flange"
        solid_output.append(f"Inferred Function: {inferred_role}")

        solid_outputs.append(solid_output)
        count += 1
        explorer.Next()

    output.append(f"Total Solids: {count}")

    # Sections summary
    sections_summary = "Sections:"
    if role_enabled:
        for role in ["Face Center", "Edge", "Corner", "Core", "Unknown"]:
            if role_counts[role] > 0:
                sections_summary += f"\n- {role}s: {role_counts[role]}"
        output.append(sections_summary)

    # Adjacency map (shared faces + proximity)
    adjacency_map = {i+1: [] for i in range(len(solids))}
    # Shared faces
    for i, s1 in enumerate(solids):
        for j, s2 in enumerate(solids):
            if i >= j:
                continue
            if solids_share_face(s1, s2):
                adjacency_map[i+1].append(j+1)
                adjacency_map[j+1].append(i+1)
    # Proximity clusters
    clusters = build_proximity_clusters(bboxes)
    output.append("Proximity Clusters (by bounding box overlap):")
    for idx, cluster in enumerate(clusters):
        members = ", ".join([f"Solid #{i+1}" for i in cluster])
        output.append(f"  Cluster {idx+1}: {members}")

    # Append adjacency info to each solid's output and add to main output
    for idx, solid_output in enumerate(solid_outputs):
        adjacents = adjacency_map.get(idx + 1, [])
        if adjacents:
            adjacents_with_roles = [f"# {i} ({solid_roles.get(i, 'Unknown')})" for i in adjacents]
            solid_output.append(f"Adjacent Solids: {', '.join(adjacents_with_roles)}")
        output.extend(solid_output)

    # Pattern/symmetry detection
    patterns = [v for v in pattern_map.values() if len(v) > 1]
    if patterns:
        output.append("\nDetected Patterns/Symmetry:")
        for p in patterns:
            output.append(f"- Pattern of {len(p)} symmetric solids: {', '.join(p)}")

    if count == 0:
        output.append("No solids found in the STEP file.")

    # Global summary block
    if count > 0:
        # Choose primary solid by max volume
        primary_idx = solid_volumes.index(max(solid_volumes))
        primary_label = solid_outputs[primary_idx][0]
        primary_holes = len(detect_holes(extract_faces(shape, unit), unit)) if solid_outputs else 0
        summary = f"""SUMMARY:
This STEP file contains {count} solids.
The primary solid ({primary_label}) has {primary_holes} holes and significant volume.
Other solids are auxiliary or repeated patterns.
Units: {unit_str}
"""
        output.insert(1, summary)

    return "\n".join(output)

def process_text_to_vstore(text, metadata):
    if not text.strip():
        return

    doc_hash = compute_hash(text)
    metadata["doc_hash"] = doc_hash

    text_splitter = NLTKTextSplitter(chunk_size=2000, chunk_overlap=100, language='english')
    text_chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata=metadata) for chunk in text_chunks]
    vstore.add_documents(documents)

def process_step(file_path, hash_set):
    try:
        if not os.path.exists(file_path):
            print(f"File no longer exists, skipping: {file_path}")
            return

        step_hash = compute_file_hash(file_path)
        if step_hash in hash_set:
            print(f"Skipping {file_path} (already processed).")
            return

        print(f" Processing: {file_path}")
        text = step_to_text(file_path)
        txt_output_path = os.path.splitext(file_path)[0] + "_summary.txt"
        with open(txt_output_path, "w", encoding="utf-8") as f:
            f.write(text)
        metadata = {"filename": os.path.basename(file_path), "file_hash": step_hash}
        process_text_to_vstore(text, metadata)

        hash_set.add(step_hash)
        save_hash_set(hash_set)
        print(f"Processing completed: {file_path}\n" + "="*100)

    except Exception as e:
        print(f"Failed to process {file_path} due to error: {e}")

def main():
    step_files = []
    for root, _, files in os.walk(step_directory):
        for file in files:
            if file.lower().endswith(('.step', '.stp')):
                step_files.append(os.path.join(root, file))

    if not step_files:
        print("No STEP files found.")
        return

    hash_set = load_hash_set()

    for file_path in tqdm(step_files, desc="Processing STEP files"):
        process_step(file_path, hash_set)

    save_hash_set(hash_set)

if __name__ == "__main__":
    main()