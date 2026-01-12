from fractions import Fraction
from math import sqrt
from numbers import Number, Complex, Real, Rational


__all__ = ["RationalComplex", "ComplexLike", "ComplexLikeRT"]


def rdiv(a: Number, b: Number) -> Number:
    if isinstance(a, int) and isinstance(b, int):
        return Fraction(a, b)
    else:
        return a / b


class RationalComplex:
    real: Rational
    imag: Rational

    def __init__(self, real: Rational = 0, imag: Rational = 0, /):
        if not isinstance(real, Rational):
            raise TypeError("RationalComplex() argument 'real'"
                            " must be a rational number")
        if not isinstance(imag, Rational):
            raise TypeError("RationalComplex() argument 'imag'"
                            " must be a rational number")
        self.real = Fraction(real)
        self.imag = Fraction(imag)

    def __repr__(self):
        if self.real == 0:
            if self.imag % 1 == 0:
                return f"{self.imag}j"
            else:
                return f"({self.imag})j"
        else:
            result = f"{self.real}"
            result += '+' if self.imag >= 0 else '-'
            if self.imag % 1 == 0:
                result += f"{abs(self.imag)}j"
            else:
                result += f"({abs(self.imag)})j"
            return result

    def __eq__(self, other: RationalComplex | Number):
        if isinstance(other, RationalComplex | complex):
            return self.real == other.real and self.imag == other.imag
        elif isinstance(other, Real):
            return self.real == other and self.imag == 0
        else:
            return NotImplemented

    def __add__(self, other: RationalComplex | Number):
        if isinstance(other, Rational):
            return RationalComplex(self.real + other, self.imag)
        elif isinstance(other, float):
            if other % 1 == 0:
                return RationalComplex(self.real + int(other), self.imag)
            else:
                return complex(self) + other
        elif isinstance(other, complex):
            if other.real % 1 == 0 and other.imag % 1 == 0:
                return RationalComplex(self.real + int(other.real), self.imag + int(other.imag))
            else:
                return complex(self) + other
        elif isinstance(other, RationalComplex):
            return RationalComplex(self.real + other.real, self.imag + other.imag)
        else:
            return NotImplemented

    def __radd__(self, other: Number):
        if isinstance(other, Rational):
            return RationalComplex(other + self.real, self.imag)
        elif isinstance(other, float):
            if other % 1 == 0:
                return RationalComplex(int(other) + self.real, self.imag)
            else:
                return other + complex(self)
        elif isinstance(other, complex):
            if other.real % 1 == 0 and other.imag % 1 == 0:
                return RationalComplex(int(other.real) + self.real, int(other.imag) + self.imag)
            else:
                return other + complex(self)
        else:
            return NotImplemented

    def __sub__(self, other: RationalComplex | Number):
        if isinstance(other, Rational):
            return RationalComplex(self.real - other, self.imag)
        elif isinstance(other, float):
            if other % 1 == 0:
                return RationalComplex(self.real - int(other), self.imag)
            else:
                return complex(self) - other
        elif isinstance(other, complex):
            if other.real % 1 == 0 and other.imag % 1 == 0:
                return RationalComplex(self.real - int(other.real), self.imag - int(other.imag))
            else:
                return complex(self) - other
        elif isinstance(other, RationalComplex):
            return RationalComplex(self.real - other.real, self.imag - other.imag)
        else:
            return NotImplemented

    def __rsub__(self, other: Number):
        if isinstance(other, Rational):
            return RationalComplex(other - self.real, -self.imag)
        elif isinstance(other, float):
            if other % 1 == 0:
                return RationalComplex(int(other) - self.real, -self.imag)
            else:
                return other - complex(self)
        elif isinstance(other, complex):
            if other.real % 1 == 0 and other.imag % 1 == 0:
                return RationalComplex(int(other.real) - self.real, int(other.imag) - self.imag)
            else:
                return other - complex(self)
        else:
            return NotImplemented

    def __mul__(self, other: RationalComplex | Number):
        if isinstance(other, Rational):
            return RationalComplex(self.real * other, self.imag * other)
        elif isinstance(other, float):
            if other % 1 == 0:
                return RationalComplex(self.real * int(other), self.imag * int(other))
            else:
                return complex(self) * other
        elif isinstance(other, complex):
            if other.real % 1 == 0 and other.imag % 1 == 0:
                return RationalComplex(self.real * int(other.real) - self.imag * int(other.imag),
                                       self.real * int(other.imag) + self.imag * int(other.real))
            else:
                return complex(self) * other
        elif isinstance(other, RationalComplex):
            return RationalComplex(self.real * other.real - self.imag * other.imag,
                                   self.real * other.imag + self.imag * other.real)
        else:
            return NotImplemented

    def __rmul__(self, other: Number):
        if isinstance(other, Rational):
            return RationalComplex(other * self.real, other * self.imag)
        elif isinstance(other, float):
            if other % 1 == 0:
                return RationalComplex(int(other) * self.real, int(other) * self.imag)
            else:
                return other * complex(self)
        elif isinstance(other, complex):
            if other.real % 1 == 0 and other.imag % 1 == 0:
                return RationalComplex(int(other.real) * self.real - int(other.imag) * self.imag,
                                       int(other.real) * self.imag + int(other.imag) * self.real)
            else:
                return other * complex(self)
        else:
            return NotImplemented

    def __truediv__(self, other: RationalComplex | Number):
        if isinstance(other, Rational):
            return RationalComplex(rdiv(self.real, other),
                                   rdiv(self.imag, other))
        elif isinstance(other, float):
            if other % 1 == 0:
                return RationalComplex(rdiv(self.real, int(other)),
                                       rdiv(self.imag, int(other)))
            else:
                return complex(self) / other
        elif isinstance(other, complex):
            if other.real % 1 == 0 and other.imag % 1 == 0:
                return RationalComplex(rdiv(self.real * int(other.real) + self.imag * int(other.imag),
                                            int(other.real) ** 2 + int(other.imag) ** 2),
                                       rdiv(self.imag * int(other.real) - self.real * int(other.imag),
                                            int(other.real) ** 2 + int(other.imag) ** 2))
            else:
                return complex(self) / other
        elif isinstance(other, RationalComplex):
            return RationalComplex(rdiv(self.real * other.real + self.imag * other.imag,
                                        other.real ** 2 + other.imag ** 2),
                                   rdiv(self.imag * other.real - self.real * other.imag,
                                        other.real ** 2 + other.imag ** 2))
        else:
            return NotImplemented

    def __rtruediv__(self, other: Number):
        if isinstance(other, Rational):
            return RationalComplex(rdiv(other * self.real,
                                        self.real ** 2 + self.imag ** 2),
                                   rdiv(-other * self.imag,
                                        self.real ** 2 + self.imag ** 2))
        elif isinstance(other, float):
            if other % 1 == 0:
                return RationalComplex(rdiv(int(other) * self.real,
                                            self.real ** 2 + self.imag ** 2),
                                       rdiv(-int(other) * self.imag,
                                            self.real ** 2 + self.imag ** 2))
            else:
                return other / complex(self)
        elif isinstance(other, complex):
            if other.real % 1 == 0 and other.imag % 1 == 0:
                return RationalComplex(rdiv(int(other.real) * self.real + int(other.imag) * self.imag,
                                            self.real ** 2 + self.imag ** 2),
                                       rdiv(int(other.imag) * self.real - int(other.real) * self.imag,
                                            self.real ** 2 + self.imag ** 2))
            else:
                return other / complex(self)
        else:
            return NotImplemented

    def __pow__(self, other: RationalComplex | Number):
        if isinstance(other, int):
            if other < 0:
                return complex(self) ** other
            elif other == 0:
                if self.real == 0 and self.imag == 0:
                    return RationalComplex(1)
                else:
                    return RationalComplex()
            else:
                result = RationalComplex(1, 0)
                for _ in range(other):
                    result *= self
                return result
        elif isinstance(other, float):
            if other % 1 == 0:
                if other < 0:
                    return complex(self) ** int(other)
                elif other == 0:
                    if self.real == 0 and self.imag == 0:
                        return RationalComplex(1)
                    else:
                        return RationalComplex()
                else:
                    result = RationalComplex(1, 0)
                    for _ in range(int(other)):
                        result *= self
                    return result
            else:
                return complex(self) ** other
        elif isinstance(other, complex):
            if other.real % 1 == 0 and other.imag == 0:
                if other.real < 0:
                    return complex(self) ** int(other.real)
                elif other.real == 0:
                    if self.real == 0 and self.imag == 0:
                        return RationalComplex(1)
                    else:
                        return RationalComplex()
                else:
                    result = RationalComplex(1, 0)
                    for _ in range(int(other.real)):
                        result *= self
                    return result
            else:
                return complex(self) ** other
        elif isinstance(other, RationalComplex):
            return complex(self) ** complex(other)
        else:
            return NotImplemented

    def __rpow__(self, other: Number):
        if isinstance(other, Number):
            if self.real % 1 == 0 and self.imag == 0:
                return other ** int(self.real)
            else:
                return other ** complex(self)
        else:
            return NotImplemented

    def __neg__(self):
        return RationalComplex(-self.real, -self.imag)

    def __pos__(self):
        return self

    def __abs__(self):
        return sqrt(self.real ** 2 + self.imag ** 2)

    def __complex__(self):
        return complex(self.real, self.imag)


type ComplexLike = Complex | RationalComplex
ComplexLikeRT = (Complex, RationalComplex)


def main():
    a = RationalComplex(3, 4)
    b = RationalComplex(1, 2)
    print(f"{a=}")
    print(f"{b=}")
    print(f"{+a=}")
    print(f"{-b=}")
    print(f"{abs(a)=}")
    print(f"{a + b=}")
    print(f"{a - b=}")
    print(f"{a * b=}")
    print(f"{a / b=}")
    print(f"{a / b * b=}")
    print(f"{b ** 2=}")


if __name__ == "__main__":
    main()
