from typing import Any


class NotFound:
    """
    Represents that something is not found.
    """

    __cond: Any

    def __init__(self, cond: Any):
        self.__cond = cond

    def __str__(self) -> str:
        if callable(self.__cond):
            return f"Not found items satisfy the condition {self.__cond}"
        else:
            return f"Not found items equal to the value {self.__cond}"

    def __repr__(self) -> str:
        return f"NotFound({repr(self.__cond)})"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
