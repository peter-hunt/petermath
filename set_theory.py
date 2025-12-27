from decimal import Decimal
from fractions import Fraction
from numbers import Number, Integral, Rational, Real, Complex


__all__ = [
    "Set",
    "ConstInterval", "EmptySet", "FiniteSet",
    "Union", "Intersection", "Difference", "Complement",
    "NumberDomain", "NaturalSet", "IntegerSet", "RationalSet", "RealSet", "ComplexeSet",
]


class Set:
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
    pass


class Union(Set):
    pass


class Intersection(Set):
    pass


class Difference(Set):
    pass


class Complement(Set):
    pass


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
        return isinstance(item, Rational | Decimal)


class RealSet(NumberDomain):
    letter: str = 'ℝ'

    def __contains__(self, item: any) -> bool:
        return isinstance(item, Real | Decimal)


class ComplexeSet(NumberDomain):
    letter: str = 'ℂ'

    def __contains__(self, item: any) -> bool:
        return isinstance(item, Complex)


def main():
    N = NaturalSet()
    Z = IntegerSet()
    Q = RationalSet()
    R = RealSet()
    C = ComplexeSet()
    print(N)
    print(isinstance(1, Integral))
    print(isinstance(1.0, Integral))
    print(isinstance(Decimal("1.0"), Integral))
    print(isinstance(Fraction(2, 1), Integral))
    print(isinstance(1.1, Complex))
    print(isinstance(Decimal("1.0"), Complex))
    print(isinstance(Fraction(2, 3), Complex))
    print(isinstance(0+0j, Complex))


if __name__ == "__main__":
    main()
