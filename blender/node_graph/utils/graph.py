"""
Created by Haonan Chang, 04/03/2022
Graph function:
- Build KNN
- Build Hierachical structure
"""
import sys
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import KDTree  # What is KDTree BTW?
import random
from itertools import compress

def spatialSampling(vert_pos, sampling_radius):
    """ Samping points with a fixed radius
        Returning sampling idx
    """
    num_left = vert_pos.shape[0]
    idx_all = list(range(num_left))
    idx_left = list(range(num_left))
    valid = [False] * num_left
    idx_sampled = list()
    while num_left > 0:
        sample_idx = random.sample(idx_left, 1)[0]  # Only sample one
        sample_vert = vert_pos[sample_idx]
        valid[sample_idx] = False
        idx_sampled.append(sample_idx)

        # Remove all vert inside sampling_radius of sample_idx
        for idx in idx_left:
            vert = vert_pos[idx]
            if np.linalg.norm(sample_vert - vert) < sampling_radius:
                valid[idx] = False
            else:
                valid[idx] = True
        idx_left = list(compress(idx_all, valid))
        num_left = len(idx_left)
    return np.array(idx_sampled, dtype=np.int32)


def buildKNN(vert_pos, knn, include_self=False, symmetrizate=False):
    """ return connection matrix
    """
    A = kneighbors_graph(vert_pos, knn, mode='connectivity', include_self=include_self)
    A = A.toarray()
    # Symmetrizate A
    if symmetrizate:
        for i in range(A.shape[0]):
            for j in range(i + 1):
                if A[i, j] != A[j, i]:
                    connect = max(A[i, j], A[j, i])
                    A[i, j] = A[j, i] = connect
    return A


def parseAdjacencyMatrix(A, knn):
    """ Parse adjacency-matrix into edge-index, (N, knn), and edge-weight (N, knn)
    """
    N = A.shape[0]
    assert(A.shape[1] == A.shape[0])
    edge_pair = np.zeros([N * knn, 2], dtype=np.int32)
    edge_idx = np.zeros([N, knn], dtype=np.int32)
    edge_weight = np.zeros([N * knn, 1], dtype=np.float32)
    for i in range(N):
        count = 0
        for j in range(N):
            if A[i, j] > 0.0:
                edge_pair[knn*i+count, 0] = i
                edge_pair[knn*i+count, 1] = j
                edge_weight[knn*i+count][0] = A[i, j]
                edge_idx[i, count] = j
                count += 1
            if count == knn:
                break
    return edge_pair, edge_weight, edge_idx
                

def sortByDistance(vert_pos, idx, nn_idx_list):
    vert = vert_pos[idx]
    distance = list()
    for nn_idx in nn_idx_list:
        distance.append(np.linalg.norm(vert - vert_pos[nn_idx]))
    return nn_idx_list[np.argsort(distance)]  # Sorted according to distance


def graphBFS(vert_pos, A_edge_base, valid, idx_level, knn, base2level, level2base):
    """ Iterative graph search:
    A_edge_base: Adjecency edge in base level
    idx_level: idx in this level
    base2level: transfer idx_base to idx_level
    level2base: transfer idx_level to idx_base
    return edges_level: edges index in this level
    """
    N = A_edge_base.shape[0]
    visited = [False] * N
    front_end = list()
    idx_base = level2base[idx_level]
    front_end.append((idx_base, 0))
    visited[idx_base] = True

    edges_level = list()
    edges_base = list()  # Used for debug
    
    last_graph_level = -1
    
    nn_idx_base_list_sorted = []
    change = False
    while len(front_end) != 0:
        nn_idx_base_list_sorted = []
        visit_idx_base, graph_level = front_end[0]
        nn_idx_base_list = A_edge_base[visit_idx_base]
        # Sorted by physical distance
        if graph_level != last_graph_level:
            nn_idx_base_list_sorted = []
            last_graph_level = graph_level
            for temp_visit_idx_base, _ in front_end:
                nn_idx_base_list_sorted.extend(sortByDistance(vert_pos, temp_visit_idx_base, A_edge_base[temp_visit_idx_base]).tolist())
            change = True
            front_end = list()
            
        for nn_idx_base in nn_idx_base_list_sorted:
            if not visited[nn_idx_base]:
                visited[nn_idx_base] = True
                front_end.append((nn_idx_base, graph_level+1))
                if valid[nn_idx_base] and change:
                    nn_idx_level = base2level[nn_idx_base]
                    edges_level.append(nn_idx_level)
                    edges_base.append(nn_idx_base)
                    if len(edges_level) >= knn:
                        return edges_level
                
        if change:
            change = False
    
    if len(edges_level) < knn:
        print("Warning: There exists node that doesn't have {knn} neighbor, filled with -1 in nn_idx...")
        return edges_level + [0] * (knn - len(edges_level))


def batchGraphBFS(vert_pos, A_edge_base, valid, idx_level_list, knn, base2level, level2base, visible=False):
    """ Run BFS in batch
    """
    edges_level_list = list()
    for idx_level in idx_level_list:
        edges_level = graphBFS(
            vert_pos, A_edge_base, valid, idx_level, knn, base2level, level2base)
        edges_level_list.append(edges_level)

    edges_pair_list = list()  # This one is used in visualization
    if visible:
        for idx, edges in zip(idx_level_list, edges_level_list):
            for edge in edges:
                edges_pair_list.append([idx, edge])
    return np.array(edges_level_list, dtype=np.int32), np.array(edges_pair_list, dtype=np.int32)


def GraphKNN(vert_pos, knn, visible=False):
    """ Run KNN in graph
    """
    A = buildKNN(vert_pos, knn)
    edge_pair, edge_weight, edge_idx = parseAdjacencyMatrix(A, knn)
    if visible:
        return edge_idx, edge_pair
    else:
        return edge_idx, np.array(list())


def buildGraphPyramid(vert_pos, A_edge_base, knn_pyramid, sample_radius_pyramid, num_level, visible=False):
    """ Generate hierachical graph based on the basical graph
    """
    graph_pyramid = dict()
    graph_pyramid["nn_index_l0"] = A_edge_base  # Initial edges
    graph_pyramid["down_sample_pos0"] = vert_pos  # Initial pos
    vert_pos_level = vert_pos

    num_vert_last = vert_pos.shape[0]

    # Down-sampling
    for level in range(1, num_level, 1):
        print(f"Down-sampling on level {level}/{num_level}...")
        sample_radius = sample_radius_pyramid[level]
        knn = knn_pyramid[level]

        idx_sample_level = spatialSampling(vert_pos_level, sample_radius)
        vert_pos_level = vert_pos_level[idx_sample_level]  # Update

        graph_pyramid[f"down_sample_idx{level}"] = idx_sample_level
        graph_pyramid[f"down_sample_pos{level}"] = vert_pos_level  # For debugging

        # Build connection
        # The based corresponding to the idx on the last level
        num_vert_level = idx_sample_level.shape[0]
        level2base = idx_sample_level
        base2level = -np.ones([num_vert_last], dtype=np.int32)
        base2level[level2base] = np.array(list(range(num_vert_level)))
        idx_level_list = list(range(num_vert_level))
        valid = np.array([False] * num_vert_last)
        valid[level2base] = True
        ## BFS version: currently used 
        edges_level_list, edges_pair_list = batchGraphBFS(
            vert_pos, A_edge_base, valid, idx_level_list, knn, base2level, level2base, visible)
        ## KNN version: disabled
        # edges_level_list, edges_pair_list = GraphKNN(vert_pos_level, knn, visible)

        graph_pyramid[f"nn_index_l{level}"] = edges_level_list
        graph_pyramid[f"nn_index_pair_l{level}"] = edges_pair_list  # For debugging

        # Update info
        num_vert_last = num_vert_level
        A_edge_base = edges_level_list

    # Up-sampling
    for level in range(num_level-1, 0, -1):
        print(f"Up-sampling on level {level}/{num_level}...")
        # Build KD-Tree of low level
        vert_pos_level = graph_pyramid[f"down_sample_pos{level}"]
        tree = KDTree(vert_pos_level, leaf_size=40)  # At which level, switch to the brute-force
        vert_pos_next_level = graph_pyramid[f"down_sample_pos{level-1}"]
        _, up_sample_idx_level = tree.query(vert_pos_next_level, k=1)  # Find the next level of sampling
        graph_pyramid[f"up_sample_idx{level}"] = up_sample_idx_level
    
    return graph_pyramid


def saveGraphPyramid(graph_pyramid, save_file):
    """ Save the graph pyramid into OcclusionFusion format as npz
    """
    np.savez(save_file,
        down_sample_idx1=graph_pyramid["down_sample_idx1"],
        down_sample_idx2=graph_pyramid["down_sample_idx2"],
        down_sample_idx3=graph_pyramid["down_sample_idx3"],
        up_sample_idx1=graph_pyramid["up_sample_idx1"],
        up_sample_idx2=graph_pyramid["up_sample_idx2"],
        up_sample_idx3=graph_pyramid["up_sample_idx3"],
        nn_index_l0=graph_pyramid["nn_index_l0"],
        nn_index_l1=graph_pyramid["nn_index_l1"],
        nn_index_l2=graph_pyramid["nn_index_l2"],
        nn_index_l3=graph_pyramid["nn_index_l3"]
    )