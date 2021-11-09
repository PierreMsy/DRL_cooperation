import numpy as np

from marl_coop.utils.sumTree import SumTree

def test_update_a_3_leaf_tree_works():
    
    memory = SumTree(3)

    memory.add(10,2)
    memory.add(20,3)
    memory.add(30,1)
    tree = memory.tree

    assert (tree.priority == 6)
    assert (tree.left_node.priority == 5)
    assert (tree.right_node.priority == 1)
    assert (tree.left_node.left_node.priority == 2)
    assert (tree.left_node.right_node.priority == 3)
    assert (tree.right_node.left_node.priority == 1)

    assert (np.all(memory.values == np.array([10,20,30])))

def test_update_a_3_leaf_tree_more_than_3_times_works():
    
    memory = SumTree(3)

    memory.add(10,2)
    memory.add(20,3)
    memory.add(30,1)
    memory.add(300,1)
    memory.add(200,2)
    tree = memory.tree

    assert (tree.priority == 4)
    assert (tree.left_node.priority == 3)
    assert (tree.right_node.priority == 1)
    assert (tree.left_node.left_node.priority == 1)
    assert (tree.left_node.right_node.priority == 2)
    assert (tree.right_node.left_node.priority == 1)

    assert (np.all(memory.values == np.array([300,200,30])))

def test_update_a_7_leaf_tree_works():
    
    memory = SumTree(7)

    memory.add(10,1)
    memory.add(20,2)
    memory.add(30,1)
    memory.add(40,4)
    memory.add(50,5)
    memory.add(60,2)
    memory.add(70,3)
    memory.add(80,1)
    memory.add(90,5)
    tree = memory.tree

    assert (tree.priority == 21)
    assert (tree.left_node.priority == 11)
    assert (tree.right_node.priority == 10)
    assert (tree.left_node.left_node.priority == 6)
    assert (tree.left_node.right_node.priority == 5)
    assert (tree.right_node.left_node.priority == 7)
    assert (tree.right_node.right_node.priority == 3)

    assert np.all(memory.values == np.array([80,90,30,40,50,60,70]))

def test_3_leaf_tree_can_be_sampled_when_one_leaf_priority_is_non_null():
    
    memory = SumTree(3)

    memory.add(17,1)
    memory.add(13,0)
    memory.add(11,0)
    
    sampled_values = np.array([memory.sample() for _ in range(3)])

    assert np.all(sampled_values == np.array([17, 17, 17]))

def test_3_leaf_tree_can_be_sampled_when_two_leaves_priority_are_non_null():
    
    memory = SumTree(3)

    memory.add(17,2)
    memory.add(13,0)
    memory.add(11,2)
    
    sampled_values = np.array([memory.sample() for _ in range(10)])

    assert np.all((sampled_values==17) | (sampled_values==11))