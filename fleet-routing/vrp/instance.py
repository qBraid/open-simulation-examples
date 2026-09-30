"""CVRP instance loading, distance matrices and solution checking.

Distances follow the TSPLIB/CVRPLIB convention for EUC_2D: Euclidean distance
rounded to the nearest integer. Known optima and best-known solutions (BKS) are
only comparable under that convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import vrplib


@dataclass
class Instance:
    name: str
    coords: np.ndarray  # (n, 2), node 0 is the depot
    demand: np.ndarray  # (n,), demand[0] == 0
    capacity: int
    dist: np.ndarray  # (n, n) integer matrix
    min_vehicles: int
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.demand)


def euc2d(coords: np.ndarray) -> np.ndarray:
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    return np.floor(d + 0.5).astype(np.int64)  # TSPLIB nint


def load(path: str) -> Instance:
    raw = vrplib.read_instance(path)
    coords = np.asarray(raw["node_coord"], dtype=float)
    demand = np.asarray(raw["demand"], dtype=np.int64)
    cap = int(raw["capacity"])
    return Instance(
        name=str(raw.get("name", path)),
        coords=coords,
        demand=demand,
        capacity=cap,
        dist=euc2d(coords),
        min_vehicles=math.ceil(demand.sum() / cap),
        meta={"comment": raw.get("comment", "")},
    )


def from_matrix(name, coords, demand, capacity, dist) -> Instance:
    demand = np.asarray(demand, dtype=np.int64)
    return Instance(
        name=name,
        coords=np.asarray(coords, dtype=float),
        demand=demand,
        capacity=int(capacity),
        dist=np.asarray(dist, dtype=np.int64),
        min_vehicles=math.ceil(demand.sum() / capacity),
    )


def route_cost(inst: Instance, routes: list[list[int]]) -> int:
    total = 0
    for r in routes:
        path = [0, *r, 0]
        total += int(sum(inst.dist[a, b] for a, b in zip(path, path[1:])))
    return total


def check(inst: Instance, routes: list[list[int]]) -> int:
    """Validate a solution (every customer once, capacity respected); return its cost."""
    seen = sorted(c for r in routes for c in r)
    if seen != list(range(1, inst.n)):
        raise ValueError("solution does not visit every customer exactly once")
    for r in routes:
        load_ = int(inst.demand[r].sum())
        if load_ > inst.capacity:
            raise ValueError(f"route over capacity: {load_} > {inst.capacity}")
    return route_cost(inst, routes)
