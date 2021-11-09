import numpy as np
from collections import deque
from math import log2, floor

class Node:

    def __init__(self, priority, index, parent=None) -> None:
        self.parent = parent
        self.left_node = None
        self.right_node = None
        self.priority = priority
        self.index = -1
#------------------    


def compute_nbr_nodes_to_create(leaf_size):
    
    def nbr_nodes(depth):
        return int((1-2**(depth+1)) / (1-2))
    
    depth = floor(log2(leaf_size))
    return  nbr_nodes(depth) + (leaf_size - 2**(depth)) * 2
    
class SumTree:

    def __init__(self, tree_size):
        self.size = tree_size
        self.values = np.zeros(tree_size, dtype=object)
        self.leaf_nodes = None
        self.tree = self._initialize_tree()
        self.write_index = 0

    def _initialize_tree(self):

        tree = Node(0, 0)
        q_nodes = deque([tree])
        leaf_nodes = deque(maxlen=self.size)
        nbr_nodes = compute_nbr_nodes_to_create(self.size)
        
        for _ in range(nbr_nodes // 2):
            
            node = q_nodes.popleft()
            node.left_node = Node(0, 0, parent=node)
            node.right_node = Node(0, 0, parent=node)
            q_nodes.append(node.left_node)
            q_nodes.append(node.right_node)
            leaf_nodes.append(node.left_node)
            leaf_nodes.append(node.right_node)
        
        if nbr_nodes % 2 == 1:
            node = q_nodes.popleft()
            node.left_node = Node(0, 0, parent=node)
            leaf_nodes.append(node.left_node)

        for index, node in enumerate(leaf_nodes):
            node.index = index
        
        self.leaf_nodes = leaf_nodes

        return tree

    def add(self, value, priority):

        self.values[self.write_index] = value
        self.update_priorities(self.leaf_nodes[self.write_index], priority)

        self.write_index = (self.write_index + 1) % self.size

    def sample(self):
        node = self.tree
        x = np.random.uniform(node.priority)

        while node.left_node is not None:
            if x <= node.left_node.priority:
                node = node.left_node
            else:
                x = x - node.left_node.priority
                node = node.right_node
        
        return self.values[node.index]


    def update_priorities(self, node, priority):
        
        def propage_update(node, delta):
            node.priority += delta
            if node.parent:
                propage_update(node.parent, delta)
        
        delta_priority = priority - node.priority
        node.priority = priority

        propage_update(node.parent, delta_priority)



