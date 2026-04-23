import csv
import heapq
from collections import deque


# FlightGraph: weighted directed graph using a dictionary of adjacency lists.
# Each key is an airport. Each value is a list of (destination, cost, duration) tuples.
#
# Time complexity:
#   add_edge       -> O(1) average
#   load_csv       -> O(E) where E is number of routes in the CSV
#   print_stats    -> O(V) where V is number of airports
# Space complexity: O(V + E)
class FlightGraph:
    def __init__(self):
        self.adj = {}

    def add_edge(self, origin, destination, cost, duration):
        if origin not in self.adj:
            self.adj[origin] = []
        if destination not in self.adj:
            self.adj[destination] = []
        self.adj[origin].append((destination, float(cost), float(duration)))

    def load_csv(self, filepath):
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.add_edge(
                    row["origin"],
                    row["destination"],
                    row["cost"],
                    row["duration_minutes"],
                )

    def print_stats(self):
        airports = len(self.adj)
        routes = sum(len(edges) for edges in self.adj.values())
        print(f"Airports: {airports}")
        print(f"Routes: {routes}")

    # Time complexity: O((V + E) log V) with a binary min-heap priority queue.
    # Space complexity: O(V) for distance map, predecessor map, and heap.
    def dijkstra(self, origin, destination, weight):
        if origin not in self.adj or destination not in self.adj:
            raise ValueError("Unknown airport.")

        if origin == destination:
            return [origin], 0.0

        if weight == "cost":
            weight_index = 1
        elif weight == "duration":
            weight_index = 2
        else:
            raise ValueError("weight must be 'cost' or 'duration'")

        distances = {origin: 0.0}
        previous = {origin: None}
        heap = [(0.0, origin)]

        while heap:
            current_distance, current_airport = heapq.heappop(heap)

            if current_distance > distances.get(current_airport, float("inf")):
                continue

            if current_airport == destination:
                break

            for edge in self.adj[current_airport]:
                neighbor = edge[0]
                edge_weight = edge[weight_index]
                new_distance = current_distance + edge_weight

                if new_distance < distances.get(neighbor, float("inf")):
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_airport
                    heapq.heappush(heap, (new_distance, neighbor))

        if destination not in distances:
            return None

        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        return path, distances[destination]

    # Time complexity: O(V + E) in the worst case, since each airport is enqueued
    # at most once and each outgoing route is examined at most once.
    # Space complexity: O(V) for the visited set and the BFS queue.
    def bfs_reachable(self, origin, max_connections):
        if origin not in self.adj:
            raise ValueError("Unknown airport.")

        if max_connections <= 0:
            return {origin}

        visited = {origin}
        queue = deque([(origin, 0)])

        while queue:
            current_airport, hops = queue.popleft()

            if hops == max_connections:
                continue

            for edge in self.adj[current_airport]:
                neighbor = edge[0]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, hops + 1))

        return visited

        # Approach (simplified brute-force):
        # 1. Build an undirected view of the graph (a -> b means a and b are linked).
        # 2. Count the number of connected components of the full graph.
        # 3. For every airport, "remove" it and recount components on the rest.
        # 4. If removing an airport increases the component count, that airport is
        #    an articulation point (its removal disconnects the graph).
        #
        # Time complexity: O(V * (V + E)) because we run a BFS for every airport.
        # Space complexity: O(V + E) for the undirected neighbor map and BFS structures.
    def find_articulation_points(self):
        undirected = {airport: set() for airport in self.adj}
        for origin, edges in self.adj.items():
            for edge in edges:
                destination = edge[0]
                undirected[origin].add(destination)
                undirected[destination].add(origin)

        def count_components(excluded):
            visited = set()
            components = 0
            for start in undirected:
                if start == excluded or start in visited:
                    continue
                components += 1
                queue = deque([start])
                visited.add(start)
                while queue:
                    node = queue.popleft()
                    for neighbor in undirected[node]:
                        if neighbor == excluded or neighbor in visited:
                            continue
                        visited.add(neighbor)
                        queue.append(neighbor)
            return components

        base_components = count_components(None)

        articulation_points = set()
        for airport in undirected:
            if count_components(airport) > base_components:
                articulation_points.add(airport)

        return articulation_points