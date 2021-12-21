import numpy as np
import pytest

from marl_coop.component.sum_tree import SumTree

def test_update_a_3_leaf_tree_works():
    '''
     6
    / \
   4   2
  / \
 3   1   
    '''
    
    memory = SumTree(3)

    memory.add(10,2)
    memory.add(20,3)
    memory.add(30,1)
    tree = memory.tree

    assert (tree.priority == 6)
    assert (tree.left_node.priority == 4)
    assert (tree.right_node.priority == 2)
    assert (tree.left_node.left_node.priority == 3)
    assert (tree.left_node.right_node.priority == 1)

    assert (np.all(memory.values == np.array([10,20,30])))

def test_update_a_3_leaf_tree_more_than_3_times_works():
    '''
     4
    / \
   3   1
  / \
 2   1   
    '''
    
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
    assert (tree.left_node.left_node.priority == 2)
    assert (tree.left_node.right_node.priority == 1)

    assert (np.all(memory.values == np.array([300,200,30])))

def test_update_a_7_leaf_tree_works():
    '''
            22
         /      \ 
      16          6
    /   \        /  \
   7     9      5    1
  / \   / \    / \
 5   2 4   5  2   3
    '''
    
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
    memory.add(00,2)
    tree = memory.tree

    assert (tree.priority == 22)
    assert (tree.left_node.priority == 16)
    assert (tree.right_node.priority == 6)
    assert (tree.left_node.left_node.priority == 7)
    assert (tree.left_node.right_node.priority == 9)
    assert (tree.right_node.left_node.priority == 5)
    assert (tree.right_node.right_node.priority == 1)

    assert np.all(memory.values == np.array([80,90,00,40,50,60,70]))

def test_3_leaf_tree_can_be_sampled_when_one_leaf_priority_is_non_null():
    
    memory = SumTree(3)

    memory.add(17,1)
    memory.add(13,0)
    memory.add(11,0)
    
    sampled_values_idx, sampled_values = memory.sample(3, replacement=True)
    sampled_values_idx = np.array(sampled_values_idx)
    sampled_values = np.array(sampled_values)

    assert np.all(sampled_values_idx == np.array([0, 0, 0]))
    assert np.all(sampled_values == np.array([17, 17, 17]))

def test_3_leaf_tree_can_be_sampled_when_two_leaves_priority_are_non_null():
    
    memory = SumTree(3)

    memory.add(17,2)
    memory.add(13,0)
    memory.add(11,2)


    sampled_values_idx, sampled_values = memory.sample(10, replacement=True)
    sampled_values_idx = np.array(sampled_values_idx)
    sampled_values = np.array(sampled_values)

    assert np.all((sampled_values==17) | (sampled_values==11))
    assert np.all((sampled_values_idx==0) | (sampled_values_idx==2))


def test_6_leaf_tree_can_be_sampled():
    '''
            9
         /      \ 
      6          3
    /   \       /  \
   4     2     3    0
  / \   / \ 
 4   0 2   0 
    '''
    
    memory = SumTree(6)

    memory.add(1,1)
    memory.add(1,2)
    memory.add(1,3)
    memory.add(1,1)
    memory.add(19,2)
    memory.add(13,0)
    memory.add(7,3)
    memory.add(5,0)
    memory.add(27,4)
    memory.add(3,0)
    tree = memory.tree

    assert (tree.priority == 9)
    assert (tree.left_node.priority == 6)
    assert (tree.right_node.priority == 3)
    assert (tree.left_node.left_node.priority == 4)
    assert (tree.left_node.right_node.priority == 2)
    assert (tree.right_node.left_node.priority == 3)
    assert (tree.right_node.right_node.priority == 0)

    sampled_values_idx, sampled_values = memory.sample(20, replacement=True)
    sampled_values_idx = np.array(sampled_values_idx)
    sampled_values = np.array(sampled_values)

    assert np.all((sampled_values_idx==4) | (sampled_values_idx==0) | (sampled_values_idx==2))
    assert np.all((sampled_values==19) | (sampled_values==7) | (sampled_values==27))


def test_4_leaf_tree_can_be_sampled_without_replacement():
    '''
           111
         /      \ 
     110          1
    /   \       /  \
   100  10     1    0
    '''
    
    memory = SumTree(4)

    memory.add(1,100) 
    memory.add(2,10)
    memory.add(3,1)
    memory.add(4,0)

    _, sampled_values = memory.sample(3, replacement=False)

    assert sampled_values  == [1, 2, 3]

def test_that_sampling_without_replacement_too_much_will_raise_exception():
    '''
           111
         /      \ 
     110          1
    /   \       /  \
   100  10     1    0
    '''
    
    memory = SumTree(4)

    memory.add(1,100)
    memory.add(2,10)
    memory.add(3,1)
    memory.add(4,0)

    _, _ = memory.sample(3, replacement=False)
    with pytest.raises(Exception):
        _, _ = memory.sample(4, replacement=False)

    