from numbers import Integral, Rational, Real, Complex

from expr import ExprLike


__all__ = [
    "Set",
    "ConstInterval", "EmptySet", "FiniteSet",
    "Union", "Intersection", "Difference", "Complement",
    "NumberDomain", "NaturalSet", "IntegerSet", "RationalSet", "RealSet", "ComplexSet",
    "NSet", "ZSet", "QSet", "RSet", "CSet",
]


class Set:
    def __init__(self, *args, **kwargs):
        raise Exception(
            "Cannot create an instance of base Set class. "
            "To use a subclass, redefine the init code."
        )

    def __contains__(self, item: any) -> bool:
        return NotImplemented


class ConstInterval(Set):
    a: Real  # lower bound
    b: Real  # upper bound
    a_inc: bool = False  # being false doesn't exclude infinity
    b_inc: bool = False  # being false doesn't exclude infinity


class EmptySet(Set):
    def __contains__(self, item: any) -> bool:
        return False


class FiniteSet(Set):
    elements: tuple[ExprLike]


class Union(Set):
    sets: list[Set]


class Intersection(Set):
    sets: list[Set]


class Difference(Set):
    left: Set
    right: Set


class Complement(Set):
    set_: Set


class NumberDomain(Set):
    letter: str

    def __repr__(self):
        return f"{type(self).__name__}()"

    def __str__(self):
        return self.letter


class NaturalSet(NumberDomain):
    letter: str = 'ℕ'

    def __contains__(self, item: any) -> bool:
        return isinstance(item, Integral) and item >= 0


class IntegerSet(NumberDomain):
    letter: str = 'ℤ'

    def __contains__(self, item: any) -> bool:
        return isinstance(item, Integral)


class RationalSet(NumberDomain):
    letter: str = 'ℚ'

    def __contains__(self, item: any) -> bool:
        return isinstance(item, Rational)


class RealSet(NumberDomain):
    letter: str = 'ℝ'

    def __contains__(self, item: any) -> bool:
        return isinstance(item, Real)


class ComplexSet(NumberDomain):
    letter: str = 'ℂ'

    def __contains__(self, item: any) -> bool:
        return isinstance(item, Complex)


NSet = NaturalSet()
ZSet = IntegerSet()
QSet = RationalSet()
RSet = RealSet()
CSet = ComplexSet()


def main():
    pass


if __name__ == "__main__":
    main()
