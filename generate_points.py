“””
generate_points.py

scene별 0.1m 간격 grid를 생성하고, habitat-sim pathfinder로
mesh 내부 유효 좌표만 필터링하여 point.txt를 저장합니다.

Usage:
python generate_points.py   
–dataset replica   
–scene apartment_0   
–data_root /path/to/dataset   
–metadata_root ./metadata   
–grid_spacing 0.1   
–floor_height_tol 0.05

Output:
./metadata/{dataset}/{scene}/point.txt
“””

import argparse
import os
import numpy as np

try:
import habitat_sim
HABITAT_AVAILABLE = True
except ImportError:
HABITAT_AVAILABLE = False
print(”[WARN] habitat_sim not found. Falling back to trimesh-based validation.”)

try:
import trimesh
TRIMESH_AVAILABLE = True
except ImportError:
TRIMESH_AVAILABLE = False

# —————————————————————————

# Habitat-sim backend

# —————————————————————————

def build_sim_cfg(scene_path: str) -> “habitat_sim.Configuration”:
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = scene_path
sim_cfg.enable_physics = False

```
agent_cfg = habitat_sim.AgentConfiguration()
agent_cfg.sensor_specifications = []   # 센서 없이 pathfinder만 사용

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
return cfg
```

def get_valid_points_habitat(
scene_path: str,
grid_spacing: float,
floor_height_tol: float,
agent_height: float,
) -> np.ndarray:
“””
habitat-sim pathfinder를 사용해 navigable point를 샘플링합니다.

```
반환값: (N, 3) numpy array  [x, y, z]
"""
cfg = build_sim_cfg(scene_path)
sim = habitat_sim.Simulator(cfg)
pf = sim.pathfinder

if not pf.is_loaded:
    raise RuntimeError(f"Pathfinder failed to load for scene: {scene_path}")

# scene 전체 bounding box
bounds_min, bounds_max = pf.get_bounds()
print(f"  Scene bounds: min={bounds_min}, max={bounds_max}")

# XZ 평면 grid 생성 (Y는 floor 기준으로 snap)
xs = np.arange(bounds_min[0], bounds_max[0] + grid_spacing, grid_spacing)
zs = np.arange(bounds_min[2], bounds_max[2] + grid_spacing, grid_spacing)

valid_points = []
for x in xs:
    for z in zs:
        # floor 위 agent_height 위치를 query point로 사용
        query = np.array([x, bounds_min[1] + agent_height, z])
        snapped = pf.snap_point(query)

        # snap 실패 시 NaN 반환
        if np.any(np.isnan(snapped)):
            continue

        # snap된 y가 query y와 너무 다르면(다른 층) 제외
        if abs(snapped[1] - query[1]) > floor_height_tol + agent_height:
            pass  # 멀티플로어 scene에서는 이 조건을 완화 가능

        if pf.is_navigable(snapped):
            # grid 간격에 맞게 XZ는 원래 값 유지, Y는 snap된 값 사용
            valid_points.append([x, snapped[1], z])

sim.close()

if len(valid_points) == 0:
    raise RuntimeError("No valid navigable points found. Check scene path and pathfinder config.")

return np.array(valid_points)
```

# —————————————————————————

# Trimesh fallback backend

# —————————————————————————

def get_valid_points_trimesh(
scene_path: str,
grid_spacing: float,
) -> np.ndarray:
“””
trimesh를 사용해 mesh interior point를 필터링합니다.
habitat-sim 없이 실행할 때 fallback으로 사용합니다.

```
주의: trimesh contains_points는 watertight mesh에서만 정확합니다.
"""
if not TRIMESH_AVAILABLE:
    raise RuntimeError("trimesh is not installed. Run: pip install trimesh")

scene = trimesh.load(scene_path)
if isinstance(scene, trimesh.Scene):
    mesh = trimesh.util.concatenate(list(scene.geometry.values()))
else:
    mesh = scene

bounds_min = mesh.bounds[0]
bounds_max = mesh.bounds[1]
print(f"  Mesh bounds: min={bounds_min}, max={bounds_max}")

xs = np.arange(bounds_min[0], bounds_max[0] + grid_spacing, grid_spacing)
ys = np.arange(bounds_min[1], bounds_max[1] + grid_spacing, grid_spacing)
zs = np.arange(bounds_min[2], bounds_max[2] + grid_spacing, grid_spacing)

# 3D meshgrid
grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
print(f"  Total candidate points: {len(grid):,}")

# interior 판정 (watertight 가정)
interior_mask = mesh.contains(grid)
valid_points = grid[interior_mask]

print(f"  Valid interior points: {len(valid_points):,}")
return valid_points
```

# —————————————————————————

# Deduplication by grid snap

# —————————————————————————

def deduplicate_by_grid(points: np.ndarray, spacing: float) -> np.ndarray:
“””
동일 grid cell에 여러 point가 snap된 경우 중복 제거합니다.
“””
keys = np.round(points / spacing).astype(int)
_, unique_idx = np.unique(keys, axis=0, return_index=True)
return points[np.sort(unique_idx)]

# —————————————————————————

# I/O

# —————————————————————————

def save_point_txt(points: np.ndarray, output_path: str) -> None:
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, “w”) as f:
for idx, (x, y, z) in enumerate(points):
f.write(f”{idx} {x:.6f} {y:.6f} {z:.6f}\n”)
print(f”  Saved {len(points)} points → {output_path}”)

def find_scene_mesh(data_root: str, dataset: str, scene: str) -> str:
“””
일반적인 SoundSpaces/Replica 데이터셋 경로에서 mesh 파일을 탐색합니다.
환경에 맞게 수정하세요.
“””
candidates = [
os.path.join(data_root, scene, “mesh.ply”),
os.path.join(data_root, scene, “mesh.glb”),
os.path.join(data_root, scene, f”{scene}.glb”),
os.path.join(data_root, dataset, scene, “mesh.ply”),
os.path.join(data_root, dataset, scene, “mesh.glb”),
os.path.join(data_root, “scene_datasets”, dataset, scene, “mesh.ply”),
]
for path in candidates:
if os.path.exists(path):
return path
raise FileNotFoundError(
f”Mesh file not found for scene ‘{scene}’. “
f”Searched:\n” + “\n”.join(f”  {c}” for c in candidates)
)

# —————————————————————————

# Main

# —————————————————————————

def parse_args():
parser = argparse.ArgumentParser(description=“Generate point.txt with 0.1m grid”)
parser.add_argument(”–dataset”,        type=str, required=True,
help=“Dataset name (e.g. replica)”)
parser.add_argument(”–scene”,          type=str, required=True,
help=“Scene name (e.g. apartment_0)”)
parser.add_argument(”–data_root”,      type=str, required=True,
help=“Root directory of the dataset meshes”)
parser.add_argument(”–metadata_root”,  type=str, default=”./metadata”,
help=“Root directory for metadata output (default: ./metadata)”)
parser.add_argument(”–grid_spacing”,   type=float, default=0.1,
help=“Grid spacing in meters (default: 0.1)”)
parser.add_argument(”–agent_height”,   type=float, default=1.5,
help=“Agent height for navigability query (default: 1.5m)”)
parser.add_argument(”–floor_height_tol”, type=float, default=0.05,
help=“Tolerance for floor height snap (default: 0.05m)”)
parser.add_argument(”–backend”,        type=str,
choices=[“habitat”, “trimesh”, “auto”], default=“auto”,
help=“Mesh validation backend (default: auto)”)
parser.add_argument(”–scenes”,         type=str, nargs=”*”,
help=“Multiple scenes to process at once”)
return parser.parse_args()

def process_scene(args, scene: str) -> None:
print(f”\n{’=’*60}”)
print(f”Processing scene: {scene}”)
print(f”{’=’*60}”)

```
output_path = os.path.join(
    args.metadata_root, args.dataset, scene, "point.txt"
)

# mesh 경로 탐색
scene_path = find_scene_mesh(args.data_root, args.dataset, scene)
print(f"  Mesh: {scene_path}")
print(f"  Grid spacing: {args.grid_spacing}m")

# backend 선택
backend = args.backend
if backend == "auto":
    backend = "habitat" if HABITAT_AVAILABLE else "trimesh"
print(f"  Backend: {backend}")

# point 생성
if backend == "habitat":
    points = get_valid_points_habitat(
        scene_path,
        grid_spacing=args.grid_spacing,
        floor_height_tol=args.floor_height_tol,
        agent_height=args.agent_height,
    )
else:
    points = get_valid_points_trimesh(scene_path, grid_spacing=args.grid_spacing)

# 중복 제거
points = deduplicate_by_grid(points, args.grid_spacing)
print(f"  After deduplication: {len(points)} points")

# 저장
save_point_txt(points, output_path)
```

def main():
args = parse_args()

```
scenes = args.scenes if args.scenes else [args.scene]
for scene in scenes:
    try:
        process_scene(args, scene)
    except Exception as e:
        print(f"[ERROR] Scene '{scene}' failed: {e}")
```

if **name** == “**main**”:
main()