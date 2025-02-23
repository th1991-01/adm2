from .actor import ProbActor, DeterActor
from .critic import Critic, VCritic

ACTOR = {
    "prob": ProbActor,
    "deter": DeterActor
}

CRITIC = {
    "q": Critic,
    "v": VCritic
}
