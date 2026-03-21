"""Graph primitives for the Flight Connection Optimizer.

This module intentionally focuses only on graph representation:
- weighted directed adjacency list
- airport/route insertion
- lightweight query helpers used by algorithms later
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterator, List, Set, Tuple


@dataclass(frozen=True)
class RouteEdge:
    """A directed flight route from one airport to another."""

    destination: str
    cost: float
    duration_minutes: float


class FlightGraph:
    """
    Weighted directed graph for flight routes.

    Internal representation:
    - _adj[airport] -> list of RouteEdge
    - _airports -> set of all airport labels seen in the graph
    """

    def __init__(self) -> None:
        self._adj: DefaultDict[str, List[RouteEdge]] = defaultdict(list)
        self._airports: Set[str] = set()

    def add_airport(self, airport: str) -> None:
        """Add an airport vertex if it does not exist yet."""
        code = airport.strip()
        if not code:
            raise ValueError("Airport name must be a non-empty string.")
        self._airports.add(code)
        # Touch adjacency so every airport has a key even with zero outgoing routes.
        _ = self._adj[code]

    def add_route(
            self,
            origin: str,
            destination: str,
            cost: float,
            duration_minutes: float,
    ) -> None:
        """
        Add a directed route origin -> destination with two weights.

        Validation:
        - airport labels must be non-empty
        - cost and duration must be non-negative
        """
        src = origin.strip()
        dst = destination.strip()
        if not src or not dst:
            raise ValueError("Origin and destination must be non-empty strings.")
        if cost < 0:
            raise ValueError("Route cost must be non-negative.")
        if duration_minutes < 0:
            raise ValueError("Route duration must be non-negative.")

        self._airports.add(src)
        self._airports.add(dst)
        self._adj[src].append(
            RouteEdge(destination=dst, cost=float(cost), duration_minutes=float(duration_minutes))
        )
        # Keep destination initialized in adjacency for consistency.
        _ = self._adj[dst]

    def has_airport(self, airport: str) -> bool:
        """Return True if airport exists in the graph."""
        return airport in self._airports

    def airports(self) -> Set[str]:
        """Return a copy of all airports."""
        return set(self._airports)

    def neighbors(self, airport: str) -> List[RouteEdge]:
        """
        Return all outgoing directed routes from an airport.

        Raises:
            KeyError: if airport does not exist in the graph.
        """
        if airport not in self._airports:
            raise KeyError(f"Unknown airport: {airport}")
        return list(self._adj[airport])

    def airport_count(self) -> int:
        """Number of airport vertices (|V|)."""
        return len(self._airports)

    def route_count(self) -> int:
        """Number of directed routes (|E|)."""
        return sum(len(edges) for edges in self._adj.values())

    def iter_routes(self) -> Iterator[Tuple[str, str, float, float]]:
        """Yield routes as (origin, destination, cost, duration_minutes)."""
        for origin, edges in self._adj.items():
            for edge in edges:
                yield (origin, edge.destination, edge.cost, edge.duration_minutes)

    def to_undirected_neighbor_sets(self) -> Dict[str, Set[str]]:
        """
        Build an undirected view of the graph.

        Useful later for connectivity or articulation-point style analysis:
        if u -> v exists, then both u is neighbor of v and v is neighbor of u.
        """
        undirected: Dict[str, Set[str]] = {airport: set() for airport in self._airports}
        for origin, edges in self._adj.items():
            for edge in edges:
                undirected[origin].add(edge.destination)
                undirected[edge.destination].add(origin)
        return undirected
