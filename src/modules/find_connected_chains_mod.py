import numpy as np

def find_connected_chains(segments, polylines):
    from collections import defaultdict, deque

    # Build adjacency map
    adjacency = defaultdict(list)
    for a, b in segments:
        adjacency[a].append(b)
        adjacency[b].append(a)

    visited = set()

    for node in adjacency:
        if node in visited:
            continue

        # Start a new chain
        chain = []
        queue = deque()
        queue.append(node)
        visited.add(node)

        while len(queue) > 0:
            current = queue.popleft()
            chain.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.appendleft(neighbor)

        # Sort the chain into an actual path (end-to-end order)
        # Find endpoints (nodes with only one neighbor)
        endpoints = [n for n in chain if len(adjacency[n]) == 1]
        if endpoints:
            # Start from one endpoint and reconstruct the path
            path = [endpoints[0]]
            seen = set(path)
            while len(path) < len(chain):
                current = path[-1]
                for neighbor in adjacency[current]:
                    if neighbor not in seen:
                        path.append(neighbor)
                        seen.add(neighbor)
                        break
            path = np.array(path)
            polylines.append(path)
        else:
            # For loops or fully connected subgraphs (unlikely), just add unordered chain
            polylines.append(chain)

    return polylines
