from expr import *


def expansion_test():
    x, y, z = symbols("xyz")
    expr = (x + 1) * (y + 2) * (z + 3)
    print(expr)
    print(expr.expand_mul())
    expr = (x + y + 3) ** 2
    print(expr)
    print(expr.expand_dist())
    expr = (x * y) ** (z + 3)
    print(expr)
    print(expr.expand_pow())
    expr = Ln(x * y * z)
    print(expr)
    print(expr.expand_log())
    expr = Ln(x ** y)
    print(expr)
    print(expr.expand_log())
    expr = (x * y * 4) ** ((z + 3) * (y + z + 4))
    print(expr)
    print(expr.expand())


def poly_test():
    x, y = symbols("xy")
    expr1 = x ** -1 + y ** y  # F, TFTF
    print(expr1)
    print(is_poly(expr1))
    print(is_poly(expr1, x))
    print(is_pos_poly(expr1, x))
    print(is_rat_poly(expr1, x))
    print(is_perfect_poly(expr1, x))
    print(split_poly(expr1))
    expr3 = 0.5 * x ** 2 + x + 1  # TTFF
    print(expr3)
    print(is_poly(expr3, x))
    print(is_pos_poly(expr3, x))
    print(is_rat_poly(expr3, x))
    print(is_perfect_poly(expr3, x))
    print(split_poly(expr3))
    expr2 = x ** 1.5 + x ** -0.5  # TFFF
    print(expr2)
    print(is_poly(expr2, x))
    print(is_pos_poly(expr2, x))
    print(is_rat_poly(expr2, x))
    print(is_perfect_poly(expr2, x))
    print(split_poly(expr2))


def derivative_test():
    x = Var('x')
    expr = sum(x ** i for i in range(5))
    print(expr)
    print(expr.diff(x))
    dexprdx = Derivative(expr, x)
    print(dexprdx)
    print(dexprdx.doit())
    print(expr)
    print(expr.diff(x, 2))
    d2exprdx2 = Derivative(expr, x, 2)
    print(d2exprdx2)
    print(d2exprdx2.doit())
    expr2 = Derivative(Cos(Sin(x)), x)
    print(expr2)
    print(expr2.doit())


def main():
    expansion_test()


if __name__ == "__main__":
    main()
