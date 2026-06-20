import numpy as np

def generate_synthetic_graph(num_nodes, node_features_dim, num_edges_factor=2):
    # Node features
    X = np.random.rand(num_nodes, node_features_dim).astype('float32')

    # Adjacency matrix (random sparse graph)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for _ in range(num_nodes * num_edges_factor):
        i, j = np.random.randint(0, num_nodes, 2)
        if i != j:
            adj[i, j] = adj[j, i] = 1

    # Ensure no self-loops for this simple GCN, though some GCNs add them implicitly
    np.fill_diagonal(adj, 0)

    # Normalize adjacency matrix (A_hat = D^-1/2 A D^-1/2)
    # Add self-loops for GCN layer if desired. For this example, let's keep it simple
    # and assume adj already contains the structure we want to model.
    # For GCN layer, usually A_hat = D^-1/2 (A + I) D^-1/2
    A_plus_I = adj + np.eye(num_nodes, dtype=np.float32)
    D = np.sum(A_plus_I, axis=1) # Degree matrix
    D_inv_sqrt = np.diag(np.power(D, -0.5))
    A_hat = D_inv_sqrt @ A_plus_I @ D_inv_sqrt

    return X, adj, A_hat