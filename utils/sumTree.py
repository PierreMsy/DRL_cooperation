import numpy as np
from collections import deque

class Node:

    def __init__(self, parent=None, priority=0):
        self.parent = parent
        self.left_node = None
        self.right_node = None
        self.priority = priority
        self.index = -1  

class SumTree:

    def __init__(self, tree_size):
        self.size = tree_size
        self.values = np.zeros(tree_size, dtype=object)
        self.leaf_nodes = None
        self.tree = self._initialize_tree()
        self.write_index = 0

    def add(self, value, priority):

        self.values[self.write_index] = value
        self.update_node_priority(self.leaf_nodes[self.write_index], priority)

        self.write_index = (self.write_index + 1) % self.size

    def sample(self, size):
        idxs, values = zip(*[self.draw() for _ in range(size)])
        return idxs, values

    def update_priorities(self, idxs, priorities):
        for idx, priority in zip(idxs, priorities):
            self.update_node_priority(self.leaf_nodes[idx], priority)

    def draw(self):

        node = self.tree
        x = np.random.uniform(node.priority)

        while node.left_node is not None:
            if x <= node.left_node.priority:
                node = node.left_node
            else:
                x = x - node.left_node.priority
                node = node.right_node
        
        return node.index, self.values[node.index]

    def update_node_priority(self, node, priority):
        
        def propage_update(node, delta):
            node.priority += delta
            if node.parent:
                propage_update(node.parent, delta)
        
        delta_priority = priority - node.priority
        node.priority = priority

        propage_update(node.parent, delta_priority)

    def _initialize_tree(self):

        tree = Node()
        q_nodes = deque([tree])
        leaf_nodes = deque(maxlen=self.size)
        nbr_nodes = 2*(self.size - 1)
        
        for _ in range(nbr_nodes // 2):
            
            node = q_nodes.popleft()
            node.left_node = Node(parent=node)
            node.right_node = Node(parent=node)
            q_nodes.append(node.left_node)
            q_nodes.append(node.right_node)
            leaf_nodes.append(node.left_node)
            leaf_nodes.append(node.right_node)

        for index, node in enumerate(leaf_nodes):
            node.index = index
        
        self.leaf_nodes = leaf_nodes

        return tree



