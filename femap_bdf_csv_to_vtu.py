#!/usr/bin/env python3
"""Convert FEMAP/Nastran BDF geometry + CSV element results to VTU/VTK.

Usage:
    python femap_bdf_csv_to_vtu.py --bdf 2.bdf --csv 2.csv --out result.vtu
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import meshio
import numpy as np

ELEMENT_TYPE_MAP = {
    "CTRIA3": ("triangle", 3),
    "CQUAD4": ("quad", 4),
    "CTETRA": ("tetra", 4),
    "CPYRAM": ("pyramid", 5),
    "CPENTA": ("wedge", 6),
    "CHEXA": ("hexahedron", 8),
}


def split_bdf_fields(line: str) -> List[str]:
    """Split a BDF line into fields (supports comma and fixed-width formats)."""
    raw = line.rstrip("\n")
    if "," in raw:
        return [chunk.strip() for chunk in raw.split(",")]
    # Nastran small-field (8 chars per field)
    return [raw[i : i + 8].strip() for i in range(0, len(raw), 8)]


def iter_cards(lines: Sequence[str]) -> List[List[str]]:
    """Group BDF physical lines into logical cards with continuations."""
    cards: List[List[str]] = []
    current: List[str] = []

    for line in lines:
        if not line.strip() or line.lstrip().startswith("$"):
            continue

        fields = split_bdf_fields(line)
        name = fields[0] if fields else ""

        is_continuation = name.startswith(("+", "*"))
        if is_continuation and current:
            current.extend(fields[1:])
            continue

        if current:
            cards.append(current)
        current = fields

    if current:
        cards.append(current)
    return cards


def _parse_float(field: str) -> float:
    text = field.strip()
    if not text:
        return 0.0
    return float(text.replace("D", "E"))


def _parse_int(field: str) -> int:
    text = field.strip()
    if not text:
        raise ValueError("Empty integer field")
    return int(text)


def read_bdf_geometry(path: Path) -> Tuple[np.ndarray, Dict[str, List[Tuple[int, List[int]]]]]:
    """Return points and element connectivity grouped by meshio cell type."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    cards = iter_cards(lines)

    node_coords: Dict[int, Tuple[float, float, float]] = {}
    elements: Dict[str, List[Tuple[int, List[int]]]] = {k: [] for k, _ in ELEMENT_TYPE_MAP.values()}

    for card in cards:
        if not card:
            continue
        name = card[0].strip().upper()

        if name == "GRID":
            # GRID, ID, CP, X, Y, Z, ...
            if len(card) < 6:
                continue
            nid = _parse_int(card[1])
            x = _parse_float(card[3])
            y = _parse_float(card[4])
            z = _parse_float(card[5])
            node_coords[nid] = (x, y, z)
            continue

        if name in ELEMENT_TYPE_MAP:
            mesh_type, n_nodes = ELEMENT_TYPE_MAP[name]
            if len(card) < 3 + n_nodes:
                continue
            eid = _parse_int(card[1])
            node_ids: List[int] = []
            for field in card[3:]:
                if not field.strip():
                    continue
                node_ids.append(_parse_int(field))
                if len(node_ids) == n_nodes:
                    break
            if len(node_ids) == n_nodes:
                elements[mesh_type].append((eid, node_ids))

    if not node_coords:
        raise ValueError(f"No GRID cards found in {path}")

    sorted_node_ids = sorted(node_coords.keys())
    node_index = {nid: idx for idx, nid in enumerate(sorted_node_ids)}
    points = np.array([node_coords[nid] for nid in sorted_node_ids], dtype=float)

    indexed_elements: Dict[str, List[Tuple[int, List[int]]]] = {}
    for cell_type, entries in elements.items():
        mapped: List[Tuple[int, List[int]]] = []
        for eid, conn in entries:
            try:
                mapped.append((eid, [node_index[nid] for nid in conn]))
            except KeyError as exc:
                raise ValueError(f"Element {eid} references missing node {exc.args[0]}") from exc
        if mapped:
            indexed_elements[cell_type] = mapped

    if not indexed_elements:
        raise ValueError(f"No supported element cards found in {path}")

    return points, indexed_elements


def read_element_results_csv(path: Path) -> Dict[int, float]:
    """Read element scalar values from CSV lines formatted as `element_id,value`."""
    results: Dict[int, float] = {}

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        row = line.strip()
        if not row or "," not in row:
            continue
        left, right, *_ = [token.strip() for token in row.split(",")]
        try:
            eid = int(left)
            value = float(right)
        except ValueError:
            continue
        results[eid] = value

    if not results:
        raise ValueError(f"No `element_id,value` rows found in {path}")

    return results


def build_mesh(points: np.ndarray, elements: Dict[str, List[Tuple[int, List[int]]]], results: Dict[int, float], result_name: str) -> meshio.Mesh:
    cells: List[meshio.CellBlock] = []
    stress_data: List[np.ndarray] = []
    eid_data: List[np.ndarray] = []

    for cell_type, entries in elements.items():
        eids = np.array([eid for eid, _ in entries], dtype=int)
        connectivity = np.array([conn for _, conn in entries], dtype=int)
        values = np.array([results.get(int(eid), math.nan) for eid in eids], dtype=float)

        cells.append(meshio.CellBlock(cell_type, connectivity))
        stress_data.append(values)
        eid_data.append(eids)

    return meshio.Mesh(
        points=points,
        cells=cells,
        cell_data={
            result_name: stress_data,
            "element_id": eid_data,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdf", required=True, type=Path, help="Input Nastran BDF file")
    parser.add_argument("--csv", required=True, type=Path, help="Input CSV with element results")
    parser.add_argument("--out", required=True, type=Path, help="Output .vtu or .vtk file")
    parser.add_argument("--result-name", default="solid_von_mises_stress", help="Cell data name for the CSV values")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points, elements = read_bdf_geometry(args.bdf)
    results = read_element_results_csv(args.csv)
    mesh = build_mesh(points, elements, results, args.result_name)
    meshio.write(args.out, mesh)

    total_cells = sum(len(block.data) for block in mesh.cells)
    matched = sum(1 for block in mesh.cell_data[args.result_name] for x in block if not np.isnan(x))
    print(f"Saved: {args.out}")
    print(f"Points: {len(points)}")
    print(f"Cells: {total_cells}")
    print(f"Results mapped: {matched}")


if __name__ == "__main__":
    main()
