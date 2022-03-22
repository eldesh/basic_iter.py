"""
This is the primitive list module.

Some basic functions on the primitive list are provided.
"""

from typing import TypeVar, List, Callable, Union, Tuple, Optional, Generic


T = TypeVar("T")
U = TypeVar("U")
S = TypeVar("S")


Predicate = Callable[[T], bool]


class NotFound(Generic[T]):
    """
    When any element are not found, 'NotFound' is returned from find family.
    'NotFound' is evaluated to 'False' constantly. However, it is possible to distinguish the searched value from a False value.

    Examples:
      >>> m = find(False, [False])
      >>> if not isinstance(m, NotFound):
      ...     print('found False')
      found False
    """

    def __init__(self, cond: Union[T, Predicate[T]]):
        self.cond = cond

    def __bool__(self) -> bool:
        """
        Evaluated to 'False' constantly.
        Then, we can check if the value is found or not in the if statement.

        Examples:
          >>> m = find(42, [1,2,3])
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


Found = Union[T, NotFound[T]]


def find(e: T, xs: List[T]) -> Found[T]:
    """
    Search a value in the list <xs> from the left to right.
    The value is equals to the value <e>.

    Returns:
      Found[T]:
        The value found in the list <xs>.
        If no values equal to the value <e>, returns <NotFound>.
    """
    if e in xs:
        return e
    return NotFound(e)


def find_if(p: Predicate[T], xs: List[T]) -> Found[T]:
    """
    Generalized <find>.
    Search a value in the list <xs> from the left to right.
    The value satisfies the predicate <p>.

    Returns:
      Found[T]:
        The value found in the list <xs>.
        If no values satisfies <p>, returns <NotFound>.

    Examples:
      >>> find (1, [1,2,3])
      1
      >>> find (42, [41,42,43])
      42
      >>> str(find (5, [0,2,4,6,8,10]))
      'Not found equals to the value 5'

    """
    for x in xs:
        if p(x):
            return x
    return NotFound(p)


def append(xs: List[T], ys: List[T]) -> List[T]:
    """
    Append xs and ys.

    Returns:
      List[T]:
        A list where <xs> is followed by <ys>.

    Examples:
      >>> xs = [1,2,3]; ys = [4,5,6]; append (xs, ys)
      [1, 2, 3, 4, 5, 6]
      >>> append ([1,2,3], [])
      [1, 2, 3]
      >>> append ([], [1,2,3])
      [1, 2, 3]
      >>> append ('hello', ' world')
      'hello world'
    """
    return xs + ys


def map(f: Callable[[T], U], xs: List[T]) -> List[U]:
    """
    Mapping all elements in the list <xs> to the result list with a mapping function <f>.

    Examples:
      >>> map (lambda x: x * 2, [1,2,3])
      [2, 4, 6]
      >>> map (str, [1,2,3])
      ['1', '2', '3']
    """
    return [f(x) for x in xs]


def filter(p: Predicate[T], xs: List[T]) -> List[T]:
    """
    Filter out elements from the list <xs> that do not satisfy the predicate <p> .

    Returns:
      List[T]:
        List of elements satisfy <p>.

    Examples:
      >>> filter (lambda x: x % 2 == 0, list(range(10)))
      [0, 2, 4, 6, 8]
    """
    ys: List[T] = []
    for x in xs:
        if p(x):
            ys.append(x)
    return ys


def null(xs: List[T]) -> bool:
    """
    Checks if the list <xs> is empty.

    Returns:
      bool: the list is empty or not.
    """
    return True if not xs else False


def reverse(xs: List[T]) -> List[T]:
    """
    Reverse the list <xs>.

    Returns:
      List[T]:
        A list of <xs> elements in reverse order.

    Examples:
      >>> reverse([1,2,3])
      [3, 2, 1]
      >>> reverse('foobarbaz')
      'zabraboof'
    """
    return xs[-1 : -len(xs) - 1 : -1]


def foldl(f: Callable[[T, U], U], e: U, xs: List[T]) -> U:
    """
    Folding left-to-right the list <xs> with the function <f> start from <e>.
    For the list <xs> is [ x0, x1, x2, ... , x(n-1), xn ],
    a calculation equivalent to the following expression is performed:
    f (xn, f (x(n-1), ... f (x2, f (x1, f (x0, e)))))

    Returns:
      U:
        A left-to-right folding of the list <xs> with the function <f>.

    Examples:
      >>> foldl (lambda x, acc: x + acc, 0, list(range(1, 11)))
      55
      >>> foldl (lambda x, acc: [x] + acc, [0], list(range(1, 6)))
      [5, 4, 3, 2, 1, 0]
      >>> foldl (lambda x, acc: [x] + [acc], [0], list(range(1, 6)))
      [5, [4, [3, [2, [1, [0]]]]]]
    """
    acc = e
    for x in xs:
        acc = f(x, acc)
    return acc


def scanl(f: Callable[[T, U], U], e: U, xs: List[T]) -> List[U]:
    """
    Folding left-to-right the list and returns a list of the intermediate values.
    For the input list <xs> and the result list are [ x0, x1, x2, ... , x(n-1), xn ] and [ r0, r1, ..., r(n-1), rn, r(n+1) ]::

      r0     is calculated from foldl f e      [  ]
      r1     is calculated from foldl f r0     [x0]
      ...
      rn     is calculated from foldl f r(n-1) [x(n-1)]
      r(n+1) is calculated from foldl f rn     [xn].

    Returns:
      List[U]:
        A list of intermediate foldl values with the function <f>.

    Examples:
      >>> scanl (lambda x, acc: x + acc, 0, list(range(1, 6)))
      [0, 1, 3, 6, 10, 15]
      >>> scanl (lambda x, acc: [x] + acc, [], list(range(1, 6)))
      [[], [1], [2, 1], [3, 2, 1], [4, 3, 2, 1], [5, 4, 3, 2, 1]]
    """
    acc = e
    res = [acc]
    for x in xs:
        acc = f(x, acc)
        res = [acc] + res
    return reverse(res)


def foldr(f: Callable[[T, U], U], e: U, xs: List[T]) -> U:
    """
    Folding right-to-left the list <xs> with the function <f> start from <e>.
    For the list <xs> is [ x0, x1, x2, ... , x(n-1), xn ],
    a calculation equivalent to the following expression is performed:
    f (x0, f (x1, ... f (x(n-2), f (x(n-1), f (xn, e)))))

    Returns:
      U:
        A right-to-left folding of the list <xs> with the function <f>.

    Example:
      Sum all values in a list.

      >>> foldr (lambda x, acc: x + acc, 100, list(range(10)))
      145

    Examples:
      >>> foldr (lambda x, acc: [x] + acc, [5], list(range(5)))
      [0, 1, 2, 3, 4, 5]
      >>> foldr (lambda x, acc: [x] + [acc], [5], list(range(5)))
      [0, [1, [2, [3, [4, [5]]]]]]
    """
    acc = e
    for x in reverse(xs):
        acc = f(x, acc)
    return acc


def scanr(f: Callable[[T, U], U], e: U, xs: List[T]) -> List[U]:
    """
    Folding right-to-left the list and returns a list of the intermediate values.
    For the input list <xs> and the result list are [ x0, x1, x2, ... , x(n-1), xn ] and [ r0, r1, ..., r(n-1), rn, r(n+1) ]::

      r0     is calculated from foldr f r1     [x0]
      r1     is calculated from foldr f r2     [x1]
      ...
      rn     is calculated from foldr f r(n+1) [xn]
      r(n+1) is calculated from foldr f e      [  ]

    Returns:
      List[U]:
        A list of intermediate foldr values with the function <f>.

    Examples:
      >>> scanr (lambda x, acc: x + acc, 5, list(range(1, 5)))
      [15, 14, 12, 9, 5]
      >>> scanr (lambda x, acc: [x] + acc, [], list(range(1, 6)))
      [[1, 2, 3, 4, 5], [2, 3, 4, 5], [3, 4, 5], [4, 5], [5], []]
      >>> scanr (max, 18, [3,6,12,4,55,11])
      [55, 55, 55, 55, 55, 18, 18]
    """
    acc = e
    res = [acc]
    for x in reverse(xs):
        acc = f(x, acc)
        res = [acc] + res
    return res


def zipWith(f: Callable[[T, U], S], xs: List[T], ys: List[U]) -> List[S]:
    """
    Returns:
      List[S]:
        A Zipped list from lists <xs> and <ys> by the function <f>.

    Raises:
      AssertionError: <xs> and <ys> have different lengths.

    Examples:
      >>> zipWith (lambda x, y: x + y, [1,2,3], [4,5,6])
      [5, 7, 9]
      >>> import re
      >>> zipWith (lambda r,p: re.search(r,p) is None
      ...         , [r'abc',r'^192',r'txt$'], ['ABC','192.168.1.1','note.txt'])
      [True, False, False]
    """
    assert len(xs) == len(ys), "required to be the same length"
    itx = xs
    ity = ys
    res: List[S] = []
    while itx:
        x = itx[0]
        y = ity[0]
        itx = itx[1:]
        ity = ity[1:]
        res = [f(x, y)] + res
    res.reverse()
    return res


def zip(xs: List[T], ys: List[U]) -> List[Tuple[T, U]]:
    """
    Equivalent to calling `zipWith` on a tuple constructor like `lambda x,y: (x,y)`.

    Raises:
      AssertionError: <xs> and <ys> have different lengths.
    """
    return zipWith(lambda x, y: (x, y), xs, ys)


def unfoldr(f: Callable[[T], Optional[Tuple[U, T]]], init: T) -> List[U]:
    res: List[U] = []
    elm = init
    while True:
        r = f(elm)
        if r:
            (x, e) = r
            res = [x] + res
            elm = e
        else:
            break
    res.reverse()
    return res


def concat(xxs: List[List[T]]) -> List[T]:
    """
    Construct a list from the list of list <xxs>.
    This is a directly concatenated list of elements of the list <xxs>.

    Returns:
      List[T]:
        A concatenated list of elements of the list <xxs>.

    Examples:
      >>> concat([[1,2,3], [4,5,6], [7,8,9]])
      [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    res: List[T] = []
    for xs in xxs:
        res += xs
    return res


def concat_map(f: Callable[[T], U], xxs: List[List[T]]) -> List[U]:
    """
    Construct a list from the list of list <xxs>.
    This is a concatenated list of elements with <f> of the list <xxs>.

    Returns:
      List[U]:
        A concatinated list of mapped elements of the list <xxs>.

    Examples:
      >>> concat_map(str, [[1,2,3], [4,5,6]])
      ['1', '2', '3', '4', '5', '6']
    """
    res: List[U] = []
    for xs in xxs:
        res += map(f, xs)
    return res


def is_prefix(xs: List[T], ys: List[T]) -> bool:
    """
    Check the list <xs> is equals to the prefix of the list <ys>.

    Examples:
      >>> is_prefix ([1,2,3], [1,2,3,4,5])
      True
      >>> is_prefix ([3,4,5], [1,2,3,4,5])
      False
      >>> is_prefix ([3,42,5], [3,42])
      False
    """
    return len(ys) >= len(xs) and ys[0 : len(xs)] == xs


def group_by(f: Callable[[T, T], bool], xs: List[T]) -> List[List[T]]:
    """
    Examples:
      >>> group_by (lambda x,y: x == y, [1,1,1,2,2,3,3,3,3])
      [[1, 1, 1], [2, 2], [3, 3, 3, 3]]
      >>> group_by (lambda x,y: x == y, [])
      []
      >>> group_by (lambda x,y: x <= y, [1,2,2,3,1,2,0,4,5,2])
      [[1, 2, 2, 3], [1, 2], [0, 4, 5], [2]]
      >>> group_by (is_prefix, ['a','ab','abra','ra','racata','racatabra'])
      [['a', 'ab', 'abra'], ['ra', 'racata', 'racatabra']]
    """
    if null(xs):
        return []

    gs: List[List[T]] = [[xs[0]]]
    for x in xs[1:]:
        if f(gs[-1][-1], x):
            gs[-1].append(x)
        else:
            gs.append([x])
    return gs


def group(xs: List[T]) -> List[List[T]]:
    """
    Equivalent to calling `group_by` on `==`.

    Examples:
      >>> group ([1,1,1,2,2,3,3,3,3])
      [[1, 1, 1], [2, 2], [3, 3, 3, 3]]
      >>> group ([])
      []
    """
    return group_by(lambda x, y: x == y, xs)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
