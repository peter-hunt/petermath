from fractions import Fraction
from itertools import zip_longest
from math import inf
from numbers import Real
from typing import Iterable

from rational_complex import RationalComplex, ComplexLike, ComplexLikeRT
from utils import cmp


__all__ = ["RealPoly", "X", "ComplexPoly", "Z"]


DIVIDE_TO_FRACTION = True
RENDER_AS_FRACTION = False
# allow RealPoly to cast real value complex numbers as coefficients
# without raising exceptions
ALLOW_REAL_CASTING = True
# whether to always print coefficient in parenthesis and no * multiplication symbol
# instead of optional * symbol and 1/-1.
# ! toggling must be done before creation, might be changed to support that
PRINT_COMPACT = True
# whether to always print zero terms for easier format to follow, works better
# with PRINT_COMPACT enabled
PRINT_ZERO_TERMS = True


if RENDER_AS_FRACTION:
    # may render 1.6666666666666667 as 7505999378950827/4503599627370496
    # if fractions are not always used through calculations
    def format_float(number: Real, /) -> str:
        return f"{number:.0f}" if number % 1 == 0 else f"{Fraction(number)}"
else:
    def format_float(number: Real, /) -> str:
        return f"{number:.0f}" if number % 1 == 0 else f"{number}"


def to_complex(number: ComplexLike, /) -> complex | RationalComplex:
    if isinstance(number, Real):
        if isinstance(number, int | Fraction):
            return RationalComplex(number)
        elif isinstance(number, float):
            if number % 1 == 0:
                return RationalComplex(int(number))
            else:
                return complex(number)
        else:
            return complex(number)
    elif isinstance(number, complex | RationalComplex):
        return number
    else:
        raise TypeError("unrecognized number type for complex conversion")


def format_complex(number: ComplexLike, /) -> tuple[bool, str]:
    number = to_complex(number)
    if number.imag == 0:
        return (cmp(number.real, 0), f"{abs(number.real)}")
    elif number.real == 0:
        return (cmp(number.imag, 0), f"{abs(number.imag)}j")
    else:
        result = f"({number.real}"
        result += '+' if number.imag > 0 else '-'
        result += f"{number.imag}j)"
        return (1, result)


def compact_complex(number: ComplexLike, /) -> str:
    number = to_complex(number)
    if number.imag == 0:
        return f"{number.real}"
    elif number.real == 0:
        return f"{number.imag}j"
    else:
        return f"{number.real}" + ('+' if number.imag > 0 else '-') + f"{abs(number.imag)}j"


if DIVIDE_TO_FRACTION:
    def div(a: Real, b: Real) -> Real:
        a = int(a) if a % 1 == 0 else a
        b = int(b) if b % 1 == 0 else b
        if b == 1:
            return a
        elif b == -1:
            return -a
        if isinstance(a, int) and isinstance(b, int):
            return a // b if a % b == 0 else Fraction(a, b)
        else:
            return a / b
else:
    def div(a: Real, b: Real) -> Real:
        return a / b
if DIVIDE_TO_FRACTION:
    def cdiv(a: ComplexLike, b: ComplexLike) -> ComplexLike:
        if isinstance(a, complex):
            if a.real % 1 == 0 and a.imag % 1 == 0:
                a = RationalComplex(int(a.real), int(a.imag))
        if isinstance(b, complex):
            if b.real % 1 == 0 and b.imag % 1 == 0:
                b = RationalComplex(int(b.real), int(b.imag))
        return a / b
else:
    def cdiv(a: ComplexLike, b: ComplexLike) -> ComplexLike:
        return a / b


class RealPoly:
    coefs: tuple[Real]

    def __init__(self, *coefs: Real):
        # constant and coeffcients of natural number power polynomial,
        # starting with constant and sorted in increasing power:
        # for example, x ** 2 - 3 * x + 2 is written as (2, -3, 1).
        # the trailing zeroes are always hidden.
        coef_list = []
        for coef in coefs:
            if ALLOW_REAL_CASTING and isinstance(coef, complex):
                if coef.imag == 0:
                    coef_list.append(coef.real)
                else:
                    raise ValueError(
                        "coefficients must be numbers with real values")
            elif isinstance(coef, Real):
                if isinstance(coef, float):
                    if coef % 1 == 0:
                        coef_list.append(int(coef))
                    else:
                        coef_list.append(coef)
                else:
                    coef_list.append(coef)
            else:
                if ALLOW_REAL_CASTING:
                    raise ValueError(
                        "coefficients must be numbers with real values")
                else:
                    raise TypeError("coefficients must be real numbers")
        self.coefs = (0,) if len(coef_list) == 0 else tuple(coef_list)
        while self.coefs[-1] == 0 and len(self.coefs) > 1:
            self.coefs = self.coefs[:-1]

    if PRINT_COMPACT:
        def __repr__(self):
            if len(self) <= 0:
                return f"({format_float(self[0])})"
            elif len(self) == 2:
                return f"({format_float(self[1])})x + ({format_float(self[0])})"
            else:
                result = f"({format_float(self[-1])})x^{len(self) - 1}"
                for expo, coef in reversed([*enumerate(self[:-1])]):
                    if coef == 0 and not PRINT_ZERO_TERMS:
                        continue
                    if expo == 0:
                        result += f" + ({format_float(coef)})"
                    elif expo == 1:
                        result += f" + ({format_float(coef)})x"
                    else:
                        result += f" + ({format_float(coef)})x^{expo}"
                return result
    else:
        def __repr__(self):
            if len(self) == 0:
                return '0'
            elif len(self) == 1:
                return format_float(self[0])
            elif len(self) == 2:
                if abs(self[1]) == 1:
                    result = 'x' if self[1] == 1 else '-x'
                else:
                    result = format_float(self[1])
                    if not self.is_int:
                        result += " * "
                    result += 'x'
                if self[0] != 0:
                    result += " + " if self[0] > 0 else " - "
                    result += format_float(abs(self[0]))
                return result
            else:
                result = '-' if self[-1] < 0 else ''
                if abs(self[-1]) == 1:
                    result += f"x^{len(self) - 1}"
                else:
                    result += format_float(abs(self[-1]))
                    if not self.is_int:
                        result += " * "
                    result += f"x^{len(self) - 1}"
                for expo, coef in reversed([*enumerate(self[:-1])]):
                    if coef == 0 and not PRINT_ZERO_TERMS:
                        continue
                    result += " + " if coef >= 0 else " - "
                    if expo == 0:
                        result += format_float(abs(coef))
                    elif expo == 1:
                        if abs(coef) != 1:
                            result += format_float(abs(coef))
                            if not self.is_int:
                                result += " * "
                        result += 'x'
                    else:
                        if abs(coef) != 1:
                            result += format_float(abs(coef))
                            if not self.is_int:
                                result += " * "
                        result += f"x^{expo}"
                return result

    def __call__(self, value: Real, /):
        return sum(coef * value ** expo for expo, coef in enumerate(self))

    def __len__(self):
        return 0 if self.coefs == (0,) else len(self.coefs)

    def __getitem__(self, key: int) -> Real:
        return self.coefs[key]

    def __iter__(self):
        return iter(self.coefs)

    def __add__(self, other: RealPoly | Real) -> RealPoly:
        if isinstance(other, RealPoly):
            return RealPoly(*(scoef + ocoef for scoef, ocoef in zip_longest(self, other, fillvalue=0)))
        elif isinstance(other, Real):
            return RealPoly(*((self[0] + other,) + self[1:]))
        else:
            return NotImplemented

    def __radd__(self, other: Real) -> RealPoly:
        if isinstance(other, Real):
            return RealPoly(*((self[0] + other,) + self[1:]))
        else:
            return NotImplemented

    def __sub__(self, other: RealPoly | Real) -> RealPoly:
        if isinstance(other, RealPoly):
            return RealPoly(*(scoef - ocoef for scoef, ocoef in zip_longest(self, other, fillvalue=0)))
        elif isinstance(other, Real):
            return RealPoly(self[0] - other, *self[1:])
        else:
            return NotImplemented

    def __rsub__(self, other: Real) -> RealPoly:
        if isinstance(other, Real):
            return RealPoly(other - self[0], *(-coef for coef in self[1:]))
        else:
            return NotImplemented

    def __mul__(self, other: RealPoly | Real) -> RealPoly:
        if isinstance(other, RealPoly):
            return RealPoly(*(
                sum(
                    scoef * other[rexpo - sexpo]
                    for sexpo, scoef in [*enumerate(self)][max(rexpo - len(other) + 1, 0):rexpo + 1]
                )
                for rexpo in range(len(self) + len(other) - 1)
            ))
        elif isinstance(other, Real):
            return RealPoly(*(coef * other for coef in self))
        else:
            return NotImplemented

    def __rmul__(self, other: Real) -> RealPoly:
        if isinstance(other, Real):
            return RealPoly(*(coef * other for coef in self))
        else:
            return NotImplemented

    def __floordiv__(self, other: RealPoly | Real) -> RealPoly:
        if isinstance(other, RealPoly):
            return divmod(self, other)[0]
        elif isinstance(other, Real):
            return divmod(self, RealPoly(other))[0]
        else:
            return NotImplemented

    def __rfloordiv__(self, other: Real) -> RealPoly:
        if isinstance(other, Real):
            return divmod(other, self)[0]
        else:
            return NotImplemented

    def __mod__(self, other: RealPoly | Real) -> RealPoly:
        if isinstance(other, RealPoly):
            return divmod(self, other)[1]
        elif isinstance(other, Real):
            return divmod(self, RealPoly(other))[1]
        else:
            return NotImplemented

    def __rmod__(self, other: Real) -> RealPoly:
        if isinstance(other, Real):
            return divmod(other, self)[1]
        else:
            return NotImplemented

    def __divmod__(self, divisor: RealPoly | Real, /) -> tuple[RealPoly, RealPoly]:
        if isinstance(divisor, Real):
            divisor = RealPoly(divisor)
        elif not isinstance(divisor, RealPoly):
            return NotImplemented
        if len(divisor) == 0:
            raise ZeroDivisionError(f"division by zero: {self} /% {divisor}")
        dend_len = len(self)
        sor_len = len(divisor)
        rounds = max(dend_len - sor_len + 1, 0)
        remainder = [*self]
        quotient = [None for _ in range(rounds)]
        for scan_expo in range(rounds - 1, -1, -1):
            coef = div(remainder.pop(), divisor[-1])
            quotient[scan_expo] = coef
            for sor_expo in range(sor_len - 1):
                dend_expo = scan_expo + sor_expo
                remainder[dend_expo] -= coef * divisor[sor_expo]
        return (RealPoly(*quotient), RealPoly() if len(remainder) == 0 else RealPoly(*remainder))

    def __rdivmod__(self, divident: Real) -> tuple[RealPoly, RealPoly]:
        return divmod(RealPoly(divident), self)

    def __pow__(self, other: RealPoly | Real) -> RealPoly:
        if isinstance(other, RealPoly):
            if len(other) > 1:
                raise ValueError(
                    "cannot raise polynomial to power of polynomial")
            expo = other[0]
        elif isinstance(other, Real):
            expo = other
        else:
            return NotImplemented

        if expo % 1 == 0:
            expo = int(expo)
        else:
            raise ValueError("cannot raise polynomial to non-integer power")

        if expo >= 0:
            result = RealPoly(1)
            for _ in range(expo):
                result *= self
            return result
        else:
            raise ValueError("cannot raise polynomial to negative power")

    def __neg__(self) -> RealPoly:
        return RealPoly(*(-coef for coef in self))

    def __pos__(self) -> RealPoly:
        return self

    @property
    def is_int(self) -> bool:
        return all(coef % 1 == 0 for coef in self)

    @property
    def derivative(self) -> RealPoly:
        return RealPoly(
            *(coef * expo
              for expo, coef in enumerate(self[1:], 1))
        )

    @property
    def integral(self) -> RealPoly:
        return RealPoly(
            0,
            *(div(coef, (expo + 1))
              for expo, coef in enumerate(self))
        )

    def get_sign_at(self, value: Real) -> int:
        if len(self) == 0:
            return 0
        elif value == -inf:
            return 1 if (self[-1] > 0) ^ (len(self) % 2 == 0) else -1
        elif value == inf:
            return 1 if self[-1] > 0 else -1
        else:
            return cmp(self(value), 0)

    def inc_mono_real_solve(self, lbound: Real, rbound: Real, /) -> Real | None:
        while True:
            mbound = (lbound + rbound) / 2
            if rbound == mbound or mbound == lbound:
                if self(lbound) == 0:
                    return lbound
                elif self(rbound) == 0:
                    return rbound
                elif self.get_sign_at(lbound) < 0 and self.get_sign_at(rbound) > 0:
                    return lbound
                else:
                    return
            msign = self.get_sign_at(mbound)
            if msign > 0:
                rbound = mbound
            elif msign < 0:
                lbound = mbound
            else:
                return mbound

    # solving within monotonic region guaranteed by either df/dx=0 bound or infinity bound
    def bound_real_solve(self, lbound: Real, rbound: Real, /) -> Real | None:
        lbound = float(lbound)
        rbound = float(rbound)
        lsign = self.get_sign_at(lbound)
        rsign = self.get_sign_at(rbound)
        if lsign == rsign and lsign in (1, -1) or lsign == 0 or rsign == 0:
            return
        if lbound == -inf and rbound == inf:
            lbound = -1
            try:
                while lbound != -inf and self.get_sign_at(lbound) != lsign:
                    lbound *= 2
            except OverflowError:
                return
            # ! return none if final sign swap takes farther than python number limit
            if lbound == -inf:
                return
        elif lbound == -inf:
            gap = 1
            try:
                while (rbound - gap) != -inf and self.get_sign_at(rbound - gap) != lsign:
                    gap *= 2
            except OverflowError:
                return
            lbound = rbound - gap
            if lbound == -inf:
                return
            if gap > 1:
                rbound = lbound + gap / 2
                if self.get_sign_at(rbound) == 0:
                    return rbound
        if rbound == inf:
            gap = 1
            try:
                while (lbound + gap) != -inf and self.get_sign_at(lbound + gap) != rsign:
                    gap *= 2
            except OverflowError:
                return
            rbound = lbound + gap
            if rbound == inf:
                return
            if gap > 1:
                lbound = rbound - gap / 2
                if self.get_sign_at(lbound) == 0:
                    return lbound
        if lsign < 0:
            return self.inc_mono_real_solve(lbound, rbound)
        else:
            return (-self).inc_mono_real_solve(lbound, rbound)

    @property
    def real_solutions(self) -> tuple[Real, ...]:
        if len(self) <= 1:
            return ()  # return none if polynomial is constant, either 0 or not
        elif len(self) == 2:
            return (-div(self[0], self[1]),)
        else:
            der_zeroes = self.derivative.real_solutions
            if len(der_zeroes) == 0:
                result = self.bound_real_solve(-inf, inf)
                return () if result is None else (result,)

            zeroes = []
            result = self.bound_real_solve(-inf, der_zeroes[0])
            if result is not None:
                zeroes.append(result)
            for i, lbound in enumerate(der_zeroes[:-1]):
                if self.get_sign_at(lbound) == 0:
                    zeroes.append(lbound)
                rbound = der_zeroes[i + 1]
                result = self.bound_real_solve(lbound, rbound)
                if result is not None:
                    zeroes.append(result)
            if self.get_sign_at(der_zeroes[-1]) == 0:
                zeroes.append(der_zeroes[-1])
            result = self.bound_real_solve(der_zeroes[-1], inf)
            if result is not None:
                zeroes.append(result)

            return tuple(zeroes)


class ComplexPoly:
    coefs: tuple[ComplexLike]

    def __init__(self, *coefs: ComplexLike):
        # constant and coeffcients of natural number power polynomial,
        # starting with constant and sorted in increasing power:
        # for example, x ** 2 - 3 * x + 2 is written as (2, -3, 1).
        # the trailing zeroes are always hidden.
        coef_list = []
        for coef in coefs:
            if isinstance(coef, complex):
                if coef.real % 1 == 0 and coef.imag % 1 == 0:
                    coef_list.append(RationalComplex(
                        int(coef.real), int(coef.imag)))
                else:
                    coef_list.append(coef)
            elif isinstance(coef, RationalComplex):
                coef_list.append(coef)
            elif isinstance(coef, Real):
                if isinstance(coef, Fraction):
                    coef_list.append(RationalComplex(coef))
                elif coef % 1 == 0:
                    coef_list.append(RationalComplex(int(coef)))
                else:
                    coef_list.append(complex(coef))
            else:
                raise ValueError("coefficients must be numbers")
        self.coefs = (0,) if len(coef_list) == 0 else tuple(coef_list)
        while self.coefs[-1] == 0 and len(self.coefs) > 1:
            self.coefs = self.coefs[:-1]

    if PRINT_COMPACT:
        def __repr__(self):
            if len(self) <= 1:
                return f"({compact_complex(self[0])})"
            elif len(self) == 2:
                return f"({compact_complex(self[1])})z + ({compact_complex(self[0])})"
            else:
                result = f"({compact_complex(self[-1])})z^{len(self) - 1}"
                for expo, coef in reversed([*enumerate(self[:-1])]):
                    if coef == 0 and not PRINT_ZERO_TERMS:
                        continue
                    elif expo == 0:
                        result += f" + ({compact_complex(coef)})"
                    elif expo == 1:
                        result += f" + ({compact_complex(coef)})z"
                    else:
                        result += f" + ({compact_complex(coef)})z^{expo}"
                return result
    else:
        def __repr__(self):
            if len(self) == 0:
                return '0'
            elif len(self) == 1:
                return f"{self[0]}"
            elif len(self) == 2:
                if self[1] == 1:
                    result = 'z'
                elif self[1] == -1:
                    result = '-z'
                else:
                    sign, cstr = format_complex(self[1])
                    result = '-' if sign == -1 else ''
                    result += f"{cstr} * z"
                if self[0] != 0:
                    sign, cstr = format_complex(self[0])
                    result += " + " if sign == 1 else " - "
                    result += cstr
                return result
            else:
                sign, cstr = format_complex(self[-1])
                result = '-' if sign == -1 else ''
                if self[-1] in (1, -1):
                    result += f"z^{len(self) - 1}"
                else:
                    result += f"{cstr} * z^{len(self) - 1}"
                for expo, coef in reversed([*enumerate(self[:-1])]):
                    if coef == 0 and not PRINT_ZERO_TERMS:
                        continue
                    sign, cstr = format_complex(coef)
                    result += " + " if sign == 1 else " - "
                    if expo == 0:
                        result += cstr
                    elif expo == 1:
                        result += 'z' if coef in (1, -1) else f"{cstr} * z"
                    else:
                        result += (f"z^{expo}" if coef in (1, -1)
                                   else f"{cstr} * z^{expo}")
                return result

    def __call__(self, value: ComplexLike, /):
        return sum(coef * value ** expo for expo, coef in enumerate(self))

    def __len__(self):
        return 0 if self.coefs == (0,) else len(self.coefs)

    def __getitem__(self, key: int) -> ComplexLike:
        return self.coefs[key]

    def __iter__(self):
        return iter(self.coefs)

    def __add__(self, other: ComplexPoly | ComplexLike) -> ComplexPoly:
        if isinstance(other, ComplexPoly):
            return ComplexPoly(*(scoef + ocoef for scoef, ocoef in zip_longest(self, other, fillvalue=0)))
        elif isinstance(other, ComplexLikeRT):
            return ComplexPoly(*((self[0] + other,) + self[1:]))
        else:
            return NotImplemented

    def __radd__(self, other: ComplexLike) -> ComplexPoly:
        if isinstance(other, ComplexLikeRT):
            return ComplexPoly(*((self[0] + other,) + self[1:]))
        else:
            return NotImplemented

    def __sub__(self, other: ComplexPoly | ComplexLike) -> ComplexPoly:
        if isinstance(other, ComplexPoly):
            return ComplexPoly(*(scoef - ocoef for scoef, ocoef in zip_longest(self, other, fillvalue=0)))
        elif isinstance(other, ComplexLikeRT):
            return ComplexPoly(self[0] - other, *self[1:])
        else:
            return NotImplemented

    def __rsub__(self, other: ComplexLike) -> ComplexPoly:
        if isinstance(other, ComplexLikeRT):
            return ComplexPoly(other - self[0], *(-coef for coef in self[1:]))
        else:
            return NotImplemented

    def __mul__(self, other: ComplexPoly | ComplexLike) -> ComplexPoly:
        if isinstance(other, ComplexPoly):
            return ComplexPoly(*(
                sum(
                    scoef * other[rexpo - sexpo]
                    for sexpo, scoef in [*enumerate(self)][max(rexpo - len(other) + 1, 0):rexpo + 1]
                )
                for rexpo in range(len(self) + len(other) - 1)
            ))
        elif isinstance(other, ComplexLikeRT):
            return ComplexPoly(*(coef * other for coef in self))
        else:
            return NotImplemented

    def __rmul__(self, other: ComplexLike) -> ComplexPoly:
        if isinstance(other, ComplexLikeRT):
            return ComplexPoly(*(coef * other for coef in self))
        else:
            return NotImplemented

    def __floordiv__(self, other: ComplexPoly | ComplexLike) -> ComplexPoly:
        if isinstance(other, ComplexPoly):
            return divmod(self, other)[0]
        elif isinstance(other, ComplexLikeRT):
            return divmod(self, ComplexPoly(other))[0]
        else:
            return NotImplemented

    def __rfloordiv__(self, other: ComplexLike) -> ComplexPoly:
        if isinstance(other, ComplexLikeRT):
            return divmod(other, self)[0]
        else:
            return NotImplemented

    def __mod__(self, other: ComplexPoly | ComplexLike) -> ComplexPoly:
        if isinstance(other, ComplexPoly):
            return divmod(self, other)[1]
        elif isinstance(other, ComplexLikeRT):
            return divmod(self, ComplexPoly(other))[1]
        else:
            return NotImplemented

    def __rmod__(self, other: ComplexLike) -> ComplexPoly:
        if isinstance(other, ComplexLikeRT):
            return divmod(other, self)[1]
        else:
            return NotImplemented

    def __divmod__(self, divisor: ComplexPoly | ComplexLike, /) -> tuple[ComplexPoly, ComplexPoly]:
        if isinstance(divisor, ComplexLikeRT):
            divisor = ComplexPoly(divisor)
        elif not isinstance(divisor, ComplexPoly):
            return NotImplemented
        if len(divisor) == 0:
            raise ZeroDivisionError(f"division by zero: {self} /% {divisor}")
        dend_len = len(self)
        sor_len = len(divisor)
        rounds = max(dend_len - sor_len + 1, 0)
        remainder = [*self]
        quotient = [None for _ in range(rounds)]
        for scan_expo in range(rounds - 1, -1, -1):
            coef = cdiv(remainder.pop(), divisor[-1])
            quotient[scan_expo] = coef
            for sor_expo in range(sor_len - 1):
                dend_expo = scan_expo + sor_expo
                remainder[dend_expo] -= coef * divisor[sor_expo]
        return (ComplexPoly(*quotient), ComplexPoly() if len(remainder) == 0 else ComplexPoly(*remainder))

    def __rdivmod__(self, divident: ComplexLike) -> tuple[ComplexPoly, ComplexPoly]:
        return divmod(ComplexPoly(divident), self)

    def __pow__(self, other: ComplexPoly | ComplexLike) -> ComplexPoly:
        if isinstance(other, ComplexPoly):
            if len(other) > 1:
                raise ValueError(
                    "cannot raise polynomial to power of polynomial")
            expo = other[0]
        elif isinstance(other, ComplexLikeRT):
            if isinstance(other, Real):
                expo = other
            else:
                if other.imag == 0:
                    expo = other.real
                else:
                    raise ValueError(
                        "cannot raise polynomial to complex power")
        else:
            return NotImplemented

        if expo % 1 == 0:
            expo = int(expo)
        else:
            raise ValueError("cannot raise polynomial to non-integer power")

        if expo >= 0:
            result = ComplexPoly(1)
            for _ in range(expo):
                result *= self
            return result
        else:
            raise ValueError("cannot raise polynomial to negative power")

    def __neg__(self) -> ComplexPoly:
        return ComplexPoly(*(-coef for coef in self))

    def __pos__(self) -> ComplexPoly:
        return self

    @property
    def derivative(self) -> ComplexPoly:
        return ComplexPoly(
            *(coef * expo
              for expo, coef in enumerate(self[1:], 1))
        )

    @property
    def integral(self) -> ComplexPoly:
        return ComplexPoly(
            0,
            *(cdiv(coef, (expo + 1))
              for expo, coef in enumerate(self))
        )


class BivariateComplexPoly:
    """
    Models the behavior of complex polynomials in terms of only real coefficients
    aka. remodeling complex number behaviors as a function,
    like p(x+yj) -> freal(x, y) + fimag(x, y)i
    For example, x + 1 would be:
    freal(x, y) = x + 1, fimag(x, y) = y
    and x^2 - x + 2 would be:
    freal(x, y) = x^2 - y^2 - x + 2, fimag(x, y) = 2xy - y
    and so on.
    # """

    def __init__(self, reals: Iterable[Iterable[Real]], imags: Iterable[Iterable[Real]]):
        self.reals = reals  # coefficients for the real part of the result
        self.imags = imags  # coefficients for the imaginary part of the result

    def __repr__(self):
        return ''

    def __call__(self, value: Real, /):
        return 0

    def __len__(self):
        return 0 if self.reals == ((0,),) and self.imags == ((0,),) else len(self.reals[0])


X = RealPoly(0, 1)
Z = ComplexPoly(0, 1)


def main():
    print(Z)
    print(ComplexPoly(0, 1, 1j, -1, -1j, 1, 1j, -1, -1j))
    a = (1+2j) * Z ** 2 + (3+4j) * Z + (5+6j)
    b = (4+3j) * Z + (2+1j)
    print(f"{a=}")
    print(f"{b=}")
    print(f"{a + b=}")
    print(f"{a - b=}")
    print(f"{a * b=}")
    print(f"{divmod(a, b)=}")
    print(f"{a // b=}")
    print(f"{a % b=}")
    print(f"{a ** 3=}")


if __name__ == "__main__":
    main()
