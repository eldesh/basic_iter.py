from typing import TypeVar, Callable, Union, Generic

T = TypeVar("T")

Predicate = Callable[[T], bool]


class NotFound(Generic[T]):
    """
    Represents that something is not found.
    It always evaluates to 'False', so unless you have searched for a value that evaluates to 'False', you can naturally inspect whether it was found or not.

    Example:
      `NotFound(42)` which is not evaluated to 'False'.

      >>> m = NotFound(42)
      >>> if not m:
      ...     print(f'not found 42')
      not found 42

    Example:
      Not found `False` itself.

      >>> m = False
      >>> if not isinstance(m, NotFound):
      ...     print('found False')
      found False
    """

    def __init__(self, cond: Union[T, Predicate[T]]):
        self.cond = cond

    def __bool__(self) -> bool:
        """
        It always evaluates to 'False'.
        Then, we can check whether the value was found or not in the if statement.

        Examples:
          >>> m = NotFound(42)
          >>> if not m:
          ...     print('not found')
          not found
        """
        return False

    def __str__(self) -> str:
        if callable(self.cond):
            return f"Not found satisfies the condition {self.cond}"
        else:
            return f"Not found equals to the value {self.cond}"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
