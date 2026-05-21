import numpy as np

# Mock 2 triangles
faces = np.array([
    [1, 2, 3],
    [2, 4, 3]
], dtype=np.int32)

e1 = faces[:, [0, 1]]
e2 = faces[:, [1, 2]]
e3 = faces[:, [2, 0]]
edges = np.vstack([e1, e2, e3])

# To find edges that appear exactly once
# We can use a structured array or convert to a 64-bit integer
# since vertex indices are < 10^7
edge_ids = edges[:, 0].astype(np.int64) * 100000000 + edges[:, 1].astype(np.int64)
# to find boundary, we check if the reverse edge exists
reverse_edge_ids = edges[:, 1].astype(np.int64) * 100000000 + edges[:, 0].astype(np.int64)

# boundary edges are those whose reverse is not in edge_ids
# We can use np.in1d
is_boundary = ~np.isin(edge_ids, reverse_edge_ids)
boundary_edges = edges[is_boundary]
print("Boundary edges:", boundary_edges.tolist())
