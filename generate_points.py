“””
generate_points.py

scene별 0.1m 간격 grid를 생성하고, 실내 유효 좌표만 필터링하여
point.txt를 저장합니다.

[핵심 수정사항]

- Habitat backend: snap 전후 XZ 거리 체크로 실외/벽 바깥 point 제거
- Trimesh backend: 3D grid 대신 XZ 2D grid에서 ray casting으로 OOM 해결

Usage:
python generate_points.py   
–dataset replica   
–scene apartment_0   
–data_root /path/to/dataset   
–metadata_root ./metadata   
–grid_spacing 0.1   
–agent_height 1.5

```
# 여러 scene:
python generate_points.py \
    --dataset replica \
    --scenes apartment_0 apartment_1 room_0 \
    --data_root /path/to/dataset
```

“””

import argparse
import os

import numpy as np

try:
import habitat_sim
HABITAT_AVAILABLE = True
except ImportError:
HABITAT_AVAILABLE = False
print(”[WARN] habitat_sim not found. Falling back to trimesh backend.”)

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
agent_cfg.sensor_specifications = []  # 센서 없이 pathfinder만 사용

return habitat_sim.Configuration(sim_cfg, [agent_cfg])
```

def get_valid_points_habitat(
scene_path: str,
grid_spacing: float,
agent_height: float,
snap_xz_tol: float,
) -> np.ndarray:
“””
Habitat-sim pathfinder 기반 실내 point 생성.

```
핵심 로직:
    query (x, floor_y, z) → snap_point() → snapped
    snapped의 XZ가 query XZ에서 snap_xz_tol 이상 벗어나면
    → query가 실외/벽 바깥이므로 제외

반환: (N, 3) [x, y, z]  y = 바닥 + agent_height
"""
cfg = build_sim_cfg(scene_path)
sim = habitat_sim.Simulator(cfg)
pf  = sim.pathfinder

if not pf.is_loaded:
    raise RuntimeError(f"Pathfinder failed to load: {scene_path}")

bounds_min, bounds_max = pf.get_bounds()
print(f"  Navmesh bounds: min={np.round(bounds_min, 3)}, max={np.round(bounds_max, 3)}")

xs = np.arange(bounds_min[0], bounds_max[0] + grid_spacing, grid_spacing)
zs = np.arange(bounds_min[2], bounds_max[2] + grid_spacing, grid_spacing)
print(f"  Grid candidates: {len(xs)} x {len(zs)} = {len(xs)*len(zs):,}")

valid_points = []

for x in xs:
    for z in zs:
        # 바닥 높이 기준으로 snap query
        query   = np.array([x, bounds_min[1], z])
        snapped = pf.snap_point(query)

        # snap 실패 (NaN)
        if np.any(np.isnan(snapped)):
            continue

        # ── 핵심 필터 ──────────────────────────────────────────────────
        # snap이 XZ를 많이 이동했다면 → query가 navmesh 바깥
        # (벽 바깥 query가 벽 안쪽으로 끌려오는 현상 차단)
        xz_shift = np.sqrt((snapped[0] - x) ** 2 + (snapped[2] - z) ** 2)
        if xz_shift > snap_xz_tol:
            continue
        # ────────────────────────────────────────────────────────────────

        if pf.is_navigable(snapped):
            # 바닥 Y에 agent_height 더해서 저장
            valid_points.append([x, snapped[1] + agent_height, z])

sim.close()

if len(valid_points) == 0:
    raise RuntimeError(
        "No valid indoor points found. "
        "snap_xz_tol을 넓히거나 grid_spacing을 조정해보세요."
    )

return np.array(valid_points)
```

# —————————————————————————

# Trimesh backend (ray casting — OOM 방지)

# —————————————————————————

def get_valid_points_trimesh(
scene_path: str,
grid_spacing: float,
agent_height: float,
ray_batch_size: int = 10_000,
) -> np.ndarray:
“””
Trimesh ray casting 기반 실내 point 생성.

```
3D meshgrid 대신 XZ 2D grid에서 위→아래로 ray를 쏘아
mesh와의 교점(바닥)을 찾습니다.
메모리 사용: O(N_xz) — 3D grid 대비 수십~수백 배 절감.

반환: (N, 3) [x, y, z]  y = 바닥 교점 + agent_height
"""
if not TRIMESH_AVAILABLE:
    raise RuntimeError("trimesh is not installed: pip install trimesh")

scene = trimesh.load(scene_path, force="mesh")
if isinstance(scene, trimesh.Scene):
    mesh = trimesh.util.concatenate(list(scene.geometry.values()))
else:
    mesh = scene

bounds_min = mesh.bounds[0]
bounds_max = mesh.bounds[1]
ray_start_y = bounds_max[1] + 1.0    # mesh 최상단 + 1m 위에서 발사
ray_dir     = np.array([0.0, -1.0, 0.0])  # 아래 방향

print(f"  Mesh bounds: min={np.round(bounds_min, 3)}, max={np.round(bounds_max, 3)}")

xs = np.arange(bounds_min[0], bounds_max[0] + grid_spacing, grid_spacing)
zs = np.arange(bounds_min[2], bounds_max[2] + grid_spacing, grid_spacing)

xg, zg = np.meshgrid(xs, zs, indexing="ij")
xg = xg.ravel()
zg = zg.ravel()
total = len(xg)
print(f"  XZ grid: {len(xs)} x {len(zs)} = {total:,} rays")

ray_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
valid_points    = []

# 배치 처리로 메모리 분산
for start in range(0, total, ray_batch_size):
    end     = min(start + ray_batch_size, total)
    batch_x = xg[start:end]
    batch_z = zg[start:end]
    n_rays  = end - start

    origins    = np.column_stack([
        batch_x,
        np.full(n_rays, ray_start_y),
        batch_z,
    ])
    directions = np.tile(ray_dir, (n_rays, 1))

    locs, ray_idx, _ = ray_intersector.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=True,
    )

    if len(locs) == 0:
        continue

    # 같은 ray 중 Y가 가장 높은 교점 = 위에서 쐈을 때 처음 만나는 바닥
    for ri in np.unique(ray_idx):
        mask    = ray_idx == ri
        floor_y = locs[mask][:, 1].max()
        valid_points.append([
            origins[ri, 0],
            floor_y + agent_height,
            origins[ri, 2],
        ])

    if (start // ray_batch_size) % 10 == 0:
        print(f"  Progress: {end:,}/{total:,} rays, "
              f"valid so far: {len(valid_points):,}")

if len(valid_points) == 0:
    raise RuntimeError("No valid floor intersections found. mesh 파일을 확인하세요.")

return np.array(valid_points)
```

# —————————————————————————

# Deduplication

# —————————————————————————

def deduplicate_by_grid(points: np.ndarray, spacing: float) -> np.ndarray:
“”“같은 grid cell에 snap된 중복 point 제거.”””
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
f”Mesh not found for ‘{scene}’.\n”
+ “\n”.join(f”  {c}” for c in candidates)
)

# —————————————————————————

# Main

# —————————————————————————

def parse_args():
parser = argparse.ArgumentParser(
description=“Generate point.txt with indoor-only points”
)
parser.add_argument(”–dataset”,        type=str, required=True)
parser.add_argument(”–scene”,          type=str, default=None)
parser.add_argument(”–scenes”,         type=str, nargs=”*”)
parser.add_argument(”–data_root”,      type=str, required=True)
parser.add_argument(”–metadata_root”,  type=str, default=”./metadata”)
parser.add_argument(”–grid_spacing”,   type=float, default=0.1,
help=“Grid spacing in meters (default: 0.1)”)
parser.add_argument(”–agent_height”,   type=float, default=1.5,
help=“Height above floor (default: 1.5m)”)
# Habitat 전용
parser.add_argument(”–snap_xz_tol”,   type=float, default=None,
help=”[Habitat] Max allowed XZ shift after snap_point. “
“기본값: grid_spacing. 크게 잡을수록 더 많은 point 포함.”)
# Trimesh 전용
parser.add_argument(”–ray_batch_size”, type=int, default=10_000,
help=”[Trimesh] Ray casting batch size (default: 10000)”)
parser.add_argument(”–backend”,        type=str,
choices=[“habitat”, “trimesh”, “auto”], default=“auto”)
return parser.parse_args()

def process_scene(args, scene: str) -> None:
print(f”\n{’=’*60}”)
print(f”Scene: {scene}”)
print(f”{’=’*60}”)

```
output_path = os.path.join(
    args.metadata_root, args.dataset, scene, "point.txt"
)
scene_path  = find_scene_mesh(args.data_root, args.dataset, scene)
print(f"  Mesh        : {scene_path}")
print(f"  Grid spacing: {args.grid_spacing}m")
print(f"  Agent height: {args.agent_height}m")

backend = args.backend
if backend == "auto":
    backend = "habitat" if HABITAT_AVAILABLE else "trimesh"
print(f"  Backend     : {backend}")

if backend == "habitat":
    snap_xz_tol = args.snap_xz_tol if args.snap_xz_tol is not None else args.grid_spacing
    print(f"  snap_xz_tol : {snap_xz_tol}m")
    points = get_valid_points_habitat(
        scene_path,
        grid_spacing=args.grid_spacing,
        agent_height=args.agent_height,
        snap_xz_tol=snap_xz_tol,
    )
else:
    points = get_valid_points_trimesh(
        scene_path,
        grid_spacing=args.grid_spacing,
        agent_height=args.agent_height,
        ray_batch_size=args.ray_batch_size,
    )

points = deduplicate_by_grid(points, args.grid_spacing)
print(f"  After dedup : {len(points):,} points")
save_point_txt(points, output_path)
```

def main():
args = parse_args()

```
scenes = args.scenes if args.scenes else ([args.scene] if args.scene else None)
if scenes is None:
    raise ValueError("--scene 또는 --scenes 를 지정하세요.")

for scene in scenes:
    try:
        process_scene(args, scene)
    except Exception as e:
        print(f"[ERROR] '{scene}': {e}")
```

if **name** == “**main**”:
main()