from algebra import *


def expansion_test():
    x, y, z = symbols("xyz")
    expr = (x + 1) * (y + 2) * (z + 3)
    print(expr)
    print(expr.expand_mul())
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
    # expansion_test()
    x, y, z = symbols("xyz")
    expr = 4 - x ** 2
    print(expr.evalf({x: -2}))
    print(expr.evalf({x: -1}))
    print(expr.evalf({x: 0}))
    print(expr.evalf({x: 1}))


if __name__ == "__main__":
    main()
