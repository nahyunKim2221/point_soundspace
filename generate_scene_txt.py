“””
generate_scene_txt.py

point.txt를 읽어, speaker 기준 0.5m 이내 listener 그룹을
JSON Lines 형식의 [scene].jsonl로 저장합니다.

포맷 (한 줄 = 딥러닝 샘플 1개):
{“speaker”: 42, “listeners”: [43, 44, 50, 51, 52]}

시뮬레이션 시 각 줄을 펼치면:
42_43.wav, 42_44.wav, 42_50.wav … (기존 wav_list 규칙 호환)

Usage:
python generate_scene_txt.py   
–dataset replica   
–scene apartment_0   
–metadata_root ./metadata   
–wav_list_root ./wav_list   
–max_dist 0.5   
–min_listeners 3   
–max_listeners 16

```
# 여러 scene 한 번에:
python generate_scene_txt.py \
    --dataset replica \
    --metadata_root ./metadata \
    --wav_list_root ./wav_list \
    --scenes apartment_0 apartment_1 room_0
```

“””

import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

# —————————————————————————

# I/O helpers

# —————————————————————————

def load_point_txt(path: str) -> tuple[np.ndarray, np.ndarray]:
“””
point.txt 파싱.

```
반환값:
    indices : (N,)   int   — 원본 index
    coords  : (N, 3) float — x y z
"""
indices, coords = [], []
with open(path, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        indices.append(int(parts[0]))
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

return np.array(indices, dtype=int), np.array(coords, dtype=float)
```

def save_jsonl(groups: list[dict], output_path: str) -> None:
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, “w”) as f:
for group in groups:
f.write(json.dumps(group, ensure_ascii=False) + “\n”)
print(f”  Saved {len(groups)} groups → {output_path}”)

# —————————————————————————

# Core grouping logic

# —————————————————————————

def build_groups(
indices: np.ndarray,
coords: np.ndarray,
max_dist: float,
min_listeners: int,
max_listeners: int,
exclude_same_point: bool = True,
) -> list[dict]:
“””
각 point를 speaker로 보고, max_dist 이내의 listener 집합을 구성합니다.

```
Parameters
----------
indices         : point.txt의 원본 인덱스 배열
coords          : (N, 3) 좌표 배열
max_dist        : speaker-listener 최대 거리 (m)
min_listeners   : 유효 그룹이 되려면 필요한 최소 listener 수
max_listeners   : 그룹당 최대 listener 수 (거리 가까운 순으로 자름)
exclude_same_point : speaker 자신을 listener에서 제외할지 여부

반환값
------
list of {"speaker": int, "listeners": list[int]}
"""
tree = cKDTree(coords)

groups = []
for i, (speaker_idx, speaker_coord) in enumerate(zip(indices, coords)):
    # max_dist 이내 모든 neighbor 탐색
    neighbor_local = tree.query_ball_point(speaker_coord, r=max_dist)

    # speaker 자신 제외
    if exclude_same_point:
        neighbor_local = [j for j in neighbor_local if j != i]

    if len(neighbor_local) < min_listeners:
        continue

    # 거리 기준 정렬 → 가까운 순으로 max_listeners개 선택
    neighbor_dists = np.linalg.norm(coords[neighbor_local] - speaker_coord, axis=1)
    sorted_order   = np.argsort(neighbor_dists)
    selected       = [neighbor_local[j] for j in sorted_order[:max_listeners]]

    listener_indices = indices[selected].tolist()

    groups.append({
        "speaker":   int(speaker_idx),
        "listeners": listener_indices,
    })

return groups
```

def build_stats(groups: list[dict]) -> dict:
“”“그룹 통계를 계산하여 출력용 dict 반환.”””
if not groups:
return {}

```
listener_counts = [len(g["listeners"]) for g in groups]
return {
    "num_groups":           len(groups),
    "total_pairs":          sum(listener_counts),
    "avg_listeners":        float(np.mean(listener_counts)),
    "min_listeners":        int(np.min(listener_counts)),
    "max_listeners":        int(np.max(listener_counts)),
    "median_listeners":     float(np.median(listener_counts)),
}
```

# —————————————————————————

# Legacy flat format helpers (기존 wav_list/*.txt 호환)

# —————————————————————————

def export_flat_wav_list(groups: list[dict], output_path: str) -> None:
“””
기존 wav_list 포맷 (0_1.wav 한 줄씩)으로도 export합니다.
jsonl과 병행 저장하여 기존 시뮬레이션 코드와의 호환성을 유지합니다.
“””
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, “w”) as f:
for group in groups:
speaker = group[“speaker”]
for listener in group[“listeners”]:
f.write(f”{speaker}_{listener}.wav\n”)
total = sum(len(g[“listeners”]) for g in groups)
print(f”  Flat wav_list: {total} pairs → {output_path}”)

# —————————————————————————

# Main

# —————————————————————————

def parse_args():
parser = argparse.ArgumentParser(
description=“Generate speaker-centric listener groups from point.txt”
)
parser.add_argument(”–dataset”,        type=str, required=True,
help=“Dataset name (e.g. replica)”)
parser.add_argument(”–scene”,          type=str, default=None,
help=“Single scene name (e.g. apartment_0)”)
parser.add_argument(”–scenes”,         type=str, nargs=”*”,
help=“Multiple scene names to process at once”)
parser.add_argument(”–metadata_root”,  type=str, default=”./metadata”,
help=“Root directory for point.txt files (default: ./metadata)”)
parser.add_argument(”–wav_list_root”,  type=str, default=”./wav_list”,
help=“Output root for .jsonl and flat .txt (default: ./wav_list)”)
parser.add_argument(”–max_dist”,       type=float, default=0.5,
help=“Max speaker-listener distance in meters (default: 0.5)”)
parser.add_argument(”–min_listeners”,  type=int, default=3,
help=“Min listeners per group (default: 3)”)
parser.add_argument(”–max_listeners”,  type=int, default=16,
help=“Max listeners per group, closest first (default: 16)”)
parser.add_argument(”–export_flat”,    action=“store_true”,
help=“Also export legacy flat wav_list .txt for compatibility”)
return parser.parse_args()

def process_scene(args, scene: str) -> None:
print(f”\n{’=’*60}”)
print(f”Processing scene: {scene}”)
print(f”{’=’*60}”)

```
# point.txt 경로
point_path = os.path.join(args.metadata_root, args.dataset, scene, "point.txt")
if not os.path.exists(point_path):
    raise FileNotFoundError(f"point.txt not found: {point_path}")

# 로드
indices, coords = load_point_txt(point_path)
print(f"  Loaded {len(indices)} points from {point_path}")

# 그룹 생성
print(f"  Grouping: max_dist={args.max_dist}m, "
      f"min_listeners={args.min_listeners}, "
      f"max_listeners={args.max_listeners}")

groups = build_groups(
    indices=indices,
    coords=coords,
    max_dist=args.max_dist,
    min_listeners=args.min_listeners,
    max_listeners=args.max_listeners,
)

# 통계 출력
stats = build_stats(groups)
if stats:
    print(f"  Groups      : {stats['num_groups']:,}")
    print(f"  Total pairs : {stats['total_pairs']:,}")
    print(f"  Listeners/group — "
          f"min: {stats['min_listeners']}, "
          f"avg: {stats['avg_listeners']:.1f}, "
          f"median: {stats['median_listeners']:.1f}, "
          f"max: {stats['max_listeners']}")
else:
    print("  [WARN] No valid groups generated. Check max_dist / min_listeners settings.")

# JSONL 저장 (메인 포맷)
jsonl_path = os.path.join(args.wav_list_root, f"{scene}.jsonl")
save_jsonl(groups, jsonl_path)

# flat txt 저장 (기존 호환용)
if args.export_flat:
    flat_path = os.path.join(args.wav_list_root, f"{scene}.txt")
    export_flat_wav_list(groups, flat_path)
```

def main():
args = parse_args()

```
# scene 목록 결정
if args.scenes:
    scenes = args.scenes
elif args.scene:
    scenes = [args.scene]
else:
    # wav_list_root 자동 탐색은 지원하지 않음
    raise ValueError("Either --scene or --scenes must be specified.")

for scene in scenes:
    try:
        process_scene(args, scene)
    except Exception as e:
        print(f"[ERROR] Scene '{scene}' failed: {e}")
```

if **name** == “**main**”:
main()