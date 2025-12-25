__all__ = [
    "Set",
    "Interval", "EmptySet", "FiniteSet",
    "Union", "Intersection", "Difference", "Complement",
    "SetBuilder",
    "NumberDomain", "NaturalSet", "IntegerSet", "RationalSet", "RealSet", "ComplexeSet",
]


class Set:
    pass


class Interval(Set):
    pass


class EmptySet(Set):
    pass


class FiniteSet(Set):
    pass


class Union(Set):
    pass


class Intersection(Set):
    pass


class Difference(Set):
    pass


class Complement(Set):
    pass


class SetBuilder(Set):
    pass


class NumberDomain(Set):
    pass


class NaturalSet(NumberDomain):
    pass


class IntegerSet(NumberDomain):
    pass


class RationalSet(NumberDomain):
    pass


class RealSet(NumberDomain):
    pass


class ComplexeSet(NumberDomain):
    pass
