from .hopper import StaticFns as HopperStaticFns
from .walker2d import StaticFns as Walker2dStaticFns
from .halfcheetah import StaticFns as HalfcheetahStaticFns

from .neorl_hopper import StaticFns as NeoRLHopperStaticFns
from .neorl_walker2d import StaticFns as NeoRLWalker2dStaticFns
from .neorl_halfcheetah import StaticFns as NeoRLHalfcheetahStaticFns

from .pen import StaticFns as PenStaticFns
from .door import StaticFns as DoorStaticFns
from .hammer import StaticFns as HammerStaticFns

from .antmaze_umaze import StaticFns as AntMazeUmazeStaticFns
from .antmaze_medium import StaticFns as AntMazeMediumStaticFns
from .antmaze_large import StaticFns as AntMazeLargeStaticFns

STATICFUNC = {
    "hopper": HopperStaticFns,
    "walker2d": Walker2dStaticFns,
    "halfcheetah": HalfcheetahStaticFns,

    "neorl-hopper": NeoRLHopperStaticFns,
    "neorl-walker2d": NeoRLWalker2dStaticFns,
    "neorl-halfcheetah": NeoRLHalfcheetahStaticFns,

    "pen": PenStaticFns,
    "door": DoorStaticFns,
    "hammer": HammerStaticFns,

    "antmaze-umaze": AntMazeUmazeStaticFns,
    "antmaze-medium": AntMazeMediumStaticFns,
    "antmaze-large": AntMazeLargeStaticFns
}
