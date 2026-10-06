from typing import Dict, List, Tuple, Any, Optional
import numpy as np

def extract_tree_edges(tree_obj: Any, parent_id: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Duyệt đệ quy cấu trúc JSON cây phân cấp trong PHEME và trả về danh sách các cạnh (parent_id, child_id).
    """
    edges = []
    if isinstance(tree_obj, dict):
        for node_id, children in tree_obj.items():
            str_node = str(node_id)
            if parent_id is not None:
                edges.append((str(parent_id), str_node))
            edges.extend(extract_tree_edges(children, parent_id=str_node))
    elif isinstance(tree_obj, list):
        for child in tree_obj:
            if isinstance(child, (dict, list)):
                edges.extend(extract_tree_edges(child, parent_id=parent_id))
            else:
                str_child = str(child)
                if parent_id is not None:
                    edges.append((str(parent_id), str_child))
    return edges

def build_tree_propagation_metrics(nodes: List[str], edges: List[Tuple[str, str]], root_id: str) -> Dict[str, Any]:
    """
    Tính toán các đặc trưng thống kê hình thái cây lan truyền (dành cho Baseline 2 - Random Forest).
    """
    adj = {n: [] for n in nodes}
    in_degree = {n: 0 for n in nodes}
    for p, c in edges:
        if p in adj and c in adj:
            adj[p].append(c)
            in_degree[c] += 1

    # Tính độ sâu từng node bằng BFS từ root
    depths = {root_id: 0}
    queue = [root_id]
    while queue:
        curr = queue.pop(0)
        for child in adj.get(curr, []):
            if child not in depths:
                depths[child] = depths[curr] + 1
                queue.append(child)

    max_depth = max(depths.values()) if depths else 0
    tree_size = len(nodes)
    leaf_nodes = sum(1 for n in nodes if len(adj[n]) == 0 and n != root_id)
    max_breadth = 0
    
    # Tính độ rộng theo từng tầng (breadth per depth level)
    depth_counts = {}
    for d in depths.values():
        depth_counts[d] = depth_counts.get(d, 0) + 1
    if depth_counts:
        max_breadth = max(depth_counts.values())

    out_degrees = [len(adj[n]) for n in nodes]
    avg_branching = float(np.mean(out_degrees)) if out_degrees else 0.0

    return {
        "tree_size": float(tree_size),
        "max_depth": float(max_depth),
        "max_breadth": float(max_breadth),
        "avg_branching": avg_branching,
        "leaf_ratio": float(leaf_nodes) / max(1.0, float(tree_size)),
    }
