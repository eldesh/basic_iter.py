from enum import Enum, auto, unique
from typing import (
    Any,
    TypeVar,
    Callable,
    Union,
    Generic,
    overload,
    cast,
)

from .not_found import NotFound
from .type import T


@unique
class Foundness(Enum):
    Found = auto()
    NotFound = auto()

    def __str__(self) -> str:
        return f"{self.name}"

    def __repr__(self) -> str:
        """
        Just name of the constructor returns, because the value of each constructor is not meaningful.

        Examples:
          >>> Foundness.Found
          Foundness.Found
          >>> Foundness.NotFound
          Foundness.NotFound
        """
        return f"Foundness.{self.name}"


class Found(Generic[T]):
    """
    Represents the direct sum of `T` and `not_found.NotFound <basic_iter.not_found.NotFound>`.
    It evaluates to True or False depending on whether it was found or not.
    This allows for a natural test to see if something was found.

    Note:
      __kind = Foundness.Found -> __value:T
      __kind = Foundness.NotFound -> __value:NotFound
    """

    __value: Union[T, NotFound]
    """The actual value.

    When its value is found, this `__value` is an instance of type `T`.
    Otherwise, it is `NotFound`.
    """

    __kind: Foundness
    """The tag of the value.

    When its value is found, this `__kind` is equals to `Foundness.Found`.
    Otherwise, it is `Foundness.NotFound`.
    """

    @classmethod
    def found(cls, value: T) -> "Found[T]":
        """
        Found value constructor.
        """
        return cls(Foundness.Found, value)

    @classmethod
    def not_found(cls, value: Any) -> "Found[T]":
        """
        NotFound value constructor.
        """
        return cls(Foundness.NotFound, NotFound(value))

    def __init__(self, kind: Foundness, value: Union[T, NotFound]):
        """
        General constructor of `Found`, it should not be used directly.
        Instead, use `found` or `not_found`.
        """
        self.__kind = kind
        self.__value = value

    def __eq__(self, other: object) -> bool:
        assert isinstance(
            other, Found
        ), "Found instances can only be compared to instances of the same type"
        return self.__kind == other.__kind and self.__value == other.__value

    def is_found(self) -> bool:
        """Checks that the value is `Found` or not."""
        return self.__kind == Foundness.Found

    def is_notfound(self) -> bool:
        """Checks that the value is `NotFound <basic_iter.not_found.NotFound>` or not."""
        return self.__kind == Foundness.NotFound

    def unwrap(self) -> T:
        """
        Unwrap a constructor and returns the body only when the value is found.
        If the value is ``NotFound``, raises `AssertionError`.

        Raises:
          AssertionError: the value is not found
        """
        if self.is_found():
            return cast(T, self.__value)

        assert False, "the value should be 'Found'"

    def __bool__(self) -> bool:
        """
        It can be used in an if statement to check if the value is found or not.
        Then always return `True` if the value is found and always return `False` if the value is `not found <basic_iter.not_found.NotFound>`.

        Examples:
          If any value is found, it evaluates to `True`.

          >>> m = Found.found(42)
          >>> if m:
          ...     print('found 42')
          found 42
          >>> m = Found.found(False)
          >>> if m:
          ...     print('found False')
          found False

        Example:
          >>> m = Found.not_found(42)
          >>> if not m:
          ...     print('not found 42')
          not found 42
        """
        return self.__kind == Foundness.Found

    def __str__(self) -> str:
        """
        Examples:
          >>> str(Found.found(42))
          'Found(42)'
        """
        return f"{str(self.__kind)}({str(self.__value)})"

    def __repr__(self) -> str:
        """
        Example:
          >>> Found.found(42)
          Found(42)

        Example:
          When __kind is :const:`Foundness.NotFound` the constructor name is omitted because __value is always :const:`NotFound`.

          >>> Found.not_found(42)
          NotFound(42)
        """
        if self.is_found():
            return f"{self.__kind.name}({repr(self.__value)})"
        else:
            return repr(self.__value)
