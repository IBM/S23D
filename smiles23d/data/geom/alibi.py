import numpy as np
from igraph import Graph


def get_shortest_path(edges):
    """
    Uses igraph dijkstra algorithm to compute the shortest path matrix
        between nodes in a graph.

    Args:
    ----
        edges (list): Edges to create the graph from

    Returns:
    -------
        np.ndarray: Shortest path matrix between nodes in the graph
    """
    weights = np.ones(len(edges))
    g = Graph(edges=edges, directed=False)
    g.es["weight"] = weights
    shortest_paths = g.distances(weights=g.es["weight"])
    shortest_path_matrix = np.array(shortest_paths) * (-1.0)
    return shortest_path_matrix


def pad_paths(shortest_paths):
    """
    Apply padding to the shortest path matrix computed from
        get_shortest_path.

    Args:
    ----
        shortest_paths (np.ndarray:): Shortest path matrix to pad.

    Returns:
    -------
        torch.tensor: Padded shortest path matrix as tensor.
    """
    max_size = max(matrix.shape[0] for matrix in shortest_paths)

    padded_matrices = []
    for matrix in shortest_paths:
        padded_matrix = np.zeros((max_size, max_size), dtype=np.float32)
        padded_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        padded_matrices.append(padded_matrix)

    tensor = np.stack(padded_matrices)
    return tensor
