from newclid.predicates.pythagoras import PythagoreanConclusions, PythagoreanPremises
from newclid.predicates.collinearity import Coll as Coll, NColl as NColl
from newclid.predicates.congruence import Cong as Cong
from newclid.predicates.midpoint import MidPoint as MidPoint
from newclid.predicates.parallelism import Para as Para, NPara as NPara
from newclid.predicates.perpendicularity import Perp as Perp, NPerp as NPerp

from newclid.predicates.cyclic import Cyclic as Cyclic
from newclid.predicates.circumcenter import Circumcenter as Circumcenter

from newclid.predicates.equal_angles import EqAngle as EqAngle
from newclid.predicates.equal_ratios import EqRatio as EqRatio, EqRatio3 as EqRatio3

from newclid.predicates.constant_length import (
    ConstantLength as ConstantLength,
    LCompute as LCompute,
)
from newclid.predicates.constant_ratio import (
    ConstantRatio as ConstantRatio,
    RCompute as RCompute,
)
from newclid.predicates.constant_angle import (
    ConstantAngle as ConstantAngle,
    ACompute as ACompute,
)

from newclid.predicates.different import Diff as Diff
from newclid.predicates.sameclock import SameClock as SameClock
from newclid.predicates.angle_types import Obtuse as Obtuse, Acute as Acute
from newclid.predicates.sameside import NSameSide as NSameSide, SameSide as SameSide
from newclid.predicates.other import Secant, ConstLine, EqPoint

from newclid.predicates.triangles_similar import (
    SimtriClock as SimtriClock,
    SimtriReflect as SimtriReflect,
)
from newclid.predicates.triangles_congruent import (
    ContriClock as ContriClock,
    ContriReflect as ContriReflect,
)
from newclid.predicates.predicate import Predicate as Predicate

SYMBOLIC_PREDICATES = (
    Coll,
    Cong,
    MidPoint,
    Para,
    Perp,
    Cyclic,
    Circumcenter,
    EqAngle,
    EqRatio,
    EqRatio3,
    ConstantLength,
    ConstantRatio,
    ConstantAngle,
    SimtriClock,
    SimtriReflect,
    ContriClock,
    ContriReflect,
    Secant,
    ConstLine,
    EqPoint,
)

NUMERICAL_PREDICATES = (
    Diff,
    NColl,
    NPara,
    NPerp,
    SameSide,
    NSameSide,
    SameClock,
    Obtuse,
    Acute,
)

INTEGRATED_PREDICATES = (PythagoreanPremises, PythagoreanConclusions)
COMPUTE = (ACompute, RCompute, LCompute)

PREDICATES = (
    SYMBOLIC_PREDICATES + NUMERICAL_PREDICATES + INTEGRATED_PREDICATES + COMPUTE
)

NAME_TO_PREDICATE = {predicate.NAME: predicate for predicate in PREDICATES}
