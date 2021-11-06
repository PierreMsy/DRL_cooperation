from abc import ABC, abstractmethod

class Base_agent(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def act(self, state):
        pass
    
    @abstractmethod
    def step(self, state, action, next_state, reward):
        pass

    @abstractmethod
    def learn(self):
        pass
