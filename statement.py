__all__ = [
    "Statement",
    "Equality",
    "Inequality", "GreaterThan", "GreaterOrEqual", "LessThan", "LessOrEqual",
    "Membership",
    "LogicNeg", "LogicOr", "LogicAnd",
]


class Statement:
    pass


class Equality(Statement):
    pass


class Inequality(Statement):
    pass


class GreaterThan(Inequality):
    pass


class GreaterOrEqual(Inequality):
    pass


class LessThan(Inequality):
    pass


class LessOrEqual(Inequality):
    pass


class Membership(Statement):
    pass


class LogicNeg(Statement):
    pass


class LogicOr(Statement):
    pass


class LogicAnd(Statement):
    pass
