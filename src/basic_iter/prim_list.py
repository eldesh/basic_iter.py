"""
This is the primitive list module.

Some basic functions on the primitive list are provided.
"""

import copy

from typing import TypeVar, List, Callable, Union, Tuple, Optional, Generic
from .not_found import NotFound


T = TypeVar("T")
U = TypeVar("U")
S = TypeVar("S")


Predicate = Callable[[T], bool]


Found = Union[T, NotFound[T]]


def last(xs: List[T]) -> T:
    """
    Extract the last element of a list, which must be non-empty.

    Raises:
      AssertionError: <xs> is not empty.
    """
    assert not (null(xs)), "must not to be empty"
    return xs[-1]


def head(xs: List[T]) -> T:
    """
    Extract the head of the list <xs>, which must to be non-empty.

    Raises:
      AssertionError: <xs> is not empty.
    """
    assert not null(xs), "must not to be empty"
    return xs[0]


def tail(xs: List[T]) -> List[T]:
    """
    Extract elements after the head of the list <xs>, which must be non-empty.

    Raises:
      AssertionError: <xs> is not empty.
    """
    assert not (null(xs)), "must not to be empty"
    return xs[1:]


def init(xs: List[T]) -> List[T]:
    """
    Extract elements except the last element of the list <xs>, which must be non-empty.

    Raises:
      AssertionError: <xs> is not empty.
    """
    assert not (null(xs)), "must not to be empty"
    return xs[0 : len(xs) - 1]


def uncons(xs: List[T]) -> Optional[Tuple[T, List[T]]]:
    """
    Decompose a list into its head and tail. If the list is empty, returns Nothing. If the list is non-empty, returns (x, xs), where x is the head of the list and xs its tail.

    Returns:
      Optional[Tuple[T, List[T]]]:
        If the list is empty, returns `None`.
        If the list is non-empty, returns `(head(x), tail(xs))`.

    Examples:
      >>> uncons([])
      >>> uncons([1,2,3])
      (1, [2, 3])
    """
    if null(xs):
        return None
    else:
        return (head(xs), tail(xs))


def null(xs: List[T]) -> bool:
    """
    Checks if the list <xs> is empty.

    Returns:
      bool: the list is empty or not.
    """
    return True if not xs else False


def find(e: T, xs: List[T]) -> Found[T]:
    """
    Search a value in the list <xs> from the left to right.
    The value is equals to the value <e>.

    Returns:
      Found[T]:
        The value found in the list <xs>.
        If no values equal to the value <e>, returns <NotFound>.

    Example:
      >>> m = find(42, [0, 1, 15, 42])
      >>> if m:
      ...     print('found 42')
      found 42

      >>> m = find(False, [False])
      >>> if not isinstance(m, NotFound):
      ...     print('found False')
      found False
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


def intersperse(e: T, xs: List[T]) -> List[T]:
    """
    Intersperses an element <e> between the elements of the list <xs>.

    Examples:
      >>> intersperse (',', [])
      []
      >>> intersperse (',', 'abcde')
      'a,b,c,d,e'
      >>> intersperse (',', 'a')
      'a'
    """
    if null(xs):
        return []

    res: List[T] = [xs[0]]
    for x in xs[1:]:
        res += [e]
        res += [x]
    if isinstance(e, str):
        return "".join(res)
    else:
        return res


def intercalate(xs: List[T], xxs: List[List[T]]) -> List[T]:
    """
    Intercalate <xs> between member lists in <xxs>.

    Examples:
      >>> intercalate(" and ", ["apple", "orange", "grape"])
      'apple and orange and grape'
    """
    if null(xxs):
        return []
    res: List[T] = copy.copy(xxs[0])
    for ys in xxs[1:]:
        res += xs
        res += ys
    if isinstance(xs, str):
        return "".join(res)
    else:
        return res


def transpose(xxs: List[List[T]]) -> List[List[T]]:
    """
    Transpose rows and columns of <xxs>.
    If some of the rows are shorter than the following rows, their elements are skipped:

    Example:
        >>> transpose ([[1,2,3],[4,5,6]])
        [[1, 4], [2, 5], [3, 6]]
        >>> transpose ([[10,11],[20],[],[30,31,32]])
        [[10, 20, 30], [11, 31], [32]]
    """
    if null(xxs):
        return []

    col = max(map(len, xxs))
    rss: List[List[T]] = []
    for i in range(col):
        # construct i-th line vector from i-th column elements
        rs: List[T] = []
        for xs in filter(lambda xs: len(xs) > i, xxs):
            rs.append(xs[i])
        rss += [rs]
    return rss


def subsequences(xs: List[T]) -> List[List[T]]:
    """
    Examples:
      >>> subsequences([])
      [[]]
      >>> subsequences("abc")
      ['', 'a', 'ab', 'abc', 'ac', 'b', 'bc', 'c']
    """

    def go(xs: List[T]) -> List[List[T]]:
        res: List[List[T]] = [[]]
        for i, x in enumerate(xs):
            res += map(lambda ys: [x] + ys, go(xs[i + 1 :]))
        return res

    res = go(xs)
    if len(res) > 1 and len(res[1]) > 0 and isinstance(res[1][0], str):
        return map(lambda xs: "".join(xs), res)
    else:
        return res


def permutations(xs: List[T]) -> List[List[T]]:
    """
    Returns:
      List[List[T]]:
        The list of permutations of the list <xs>.

    Examples:
      >>> permutations([1,2,3])
      [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    """
    if null(xs):
        return [[]]

    res: List[List[T]] = []
    ys: List[T] = copy.copy(xs)
    for i in range(len(xs)):
        y = ys.pop(i)
        res += map(lambda xs: [y] + xs, permutations(ys))
        ys.insert(i, y)
    return res


def foldl(f: Callable[[U, T], U], e: U, xs: List[T]) -> U:
    """
    Folding left-to-right the list <xs> with the function <f> start from <e>.
    For the list <xs> is [ x0, x1, x2, ... , x(n-1), xn ],
    a calculation equivalent to the following expression is performed:
    f (xn, f (x(n-1), ... f (x2, f (x1, f (x0, e)))))

    Returns:
      U:
        A left-to-right folding of the list <xs> with the function <f>.

    Examples:
      >>> foldl (lambda acc, x: x + acc, 0, list(range(1, 11)))
      55
      >>> foldl (lambda acc, x: [x] + acc, [0], list(range(1, 6)))
      [5, 4, 3, 2, 1, 0]
      >>> foldl (lambda acc, x: [x] + [acc], [0], list(range(1, 6)))
      [5, [4, [3, [2, [1, [0]]]]]]
    """
    acc = e
    for x in xs:
        acc = f(acc, x)
    return acc


def foldl1(f: Callable[[U, T], U], xs: List[T]) -> U:
    """
    A variant of <foldl> that has no base case.
    Thus, this may only be applied to non-empty lists.

    Returns:
      U:
        A left-to-right folding of the list <xs> with the function <f>.

    Raises:
      AssertionError: <xs> is empty.

    Examples:
      >>> foldl1 (lambda acc, x: x + acc, list(range(1, 11)))
      55
      >>> foldl1 (lambda acc, x: [x] + ([acc] if 1 == acc else acc), list(range(1, 6)))
      [5, 4, 3, 2, 1]
      >>> foldl1 (lambda acc, x: [x] + [acc], list(range(1, 6)))
      [5, [4, [3, [2, 1]]]]
    """
    assert xs, "required to be a non-empty list"
    return foldl(f, xs[0], xs[1:])


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


def foldr1(f: Callable[[T, U], U], xs: List[T]) -> U:
    """
    Folding right-to-left the non-empty list <xs> with the function <f>.
    For the list <xs> is [ x0, x1, x2, ... , x(n-1), xn ],
    a calculation equivalent to the following expression is performed:
    f (x0, f (x1, ... f (x(n-2), f (x(n-1), f (xn, e)))))

    Returns:
      U:
        A right-to-left folding of the list <xs> with the function <f>.

    Raises:
      AssertionError: <xs> is empty.

    Example:
      Sum all values in a list.

      >>> foldr1 (lambda x, acc: x + acc, list(range(10)))
      45

    Examples:
      >>> foldr1 (lambda x, acc: [x] + [acc], list(range(5)))
      [0, [1, [2, [3, 4]]]]
    """
    assert xs, "required to be a non-empty list"
    return foldr(f, last(xs), init(xs))


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


def concat_map(f: Callable[[T], List[U]], xs: List[T]) -> List[U]:
    """
    Map the function <f> over all elements of the list <xs>, and concatenate the resulting lists.

    Returns:
      List[U]:
        A concatinated list of mapped elements of the list <xs>.

    Examples:
      >>> concat_map(lambda x: [x, x+1], [1,2,3,4,5])
      [1, 2, 2, 3, 3, 4, 4, 5, 5, 6]
    """
    res: List[U] = []
    for x in xs:
        res += f(x)
    return res


def and_list(xs: List[bool]) -> bool:
    """
    Returns:
      bool:
        Whether the all elements of the list <xs> is True.
    """
    for x in xs:
        if not x:
            return False
    return True


def or_list(xs: List[bool]) -> bool:
    """
    Returns:
      bool:
        Whether any elements of the list <xs> is True.
    """
    for x in xs:
        if x:
            return True
    return False


def any(p: Predicate[T], xs: List[T]) -> bool:
    """
    Check any elements of the list <xs> satisfy the predicate <p>.

    Examples:
      >>> any(lambda x: x % 2 == 0, [1,3,5,7,9])
      False
      >>> import re
      >>> any(lambda x: re.search(r'a*b{2}c', x), ['bc', 'ab', 'aaabbc'])
      True
    """
    for x in xs:
        if p(x):
            return True
    return False


def all(p: Predicate[T], xs: List[T]) -> bool:
    """
    Check all elements of the list <xs> satisfy the predicate <p>.

    Examples:
      >>> all(lambda x: x % 2 == 0, [0,2,4,6,7])
      False
      >>> import re
      >>> all(lambda x: re.search(r'a*b', x), ['abc', 'b', 'aaab'])
      True
    """
    for x in xs:
        if not (p(x)):
            return False
    return True


def scanl(f: Callable[[U, T], U], e: U, xs: List[T]) -> List[U]:
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
      >>> scanl (lambda acc, x: x + acc, 0, list(range(1, 6)))
      [0, 1, 3, 6, 10, 15]
      >>> scanl (lambda acc, x: [x] + acc, [], list(range(1, 6)))
      [[], [1], [2, 1], [3, 2, 1], [4, 3, 2, 1], [5, 4, 3, 2, 1]]
    """
    acc = e
    res = [acc]
    for x in xs:
        acc = f(acc, x)
        res = [acc] + res
    return reverse(res)


def scanl1(f: Callable[[U, T], U], xs: List[T]) -> List[U]:
    """
    scanl1 is a variant of scanl that has no starting value argument

    Raises:
      AssertionError: <xs> is empty.
    """
    assert xs, "required to be a non-empty list"
    return scanl(f, xs[0], xs[1:])


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


def mapAccumL(f: Callable[[T, U], Tuple[T, S]], e: T, xs: List[U]) -> Tuple[T, List[S]]:
    """
    mapAccumL transforms the list <xs> with <f> and simultaneously accumulates its elements from left-to-right into a value of <T>.

    Returns:
      Tuple[T, List[S]]:
        A tuple of the accumulated value and the transformed list.

    Examples:
      >>> mapAccumL(lambda acc, x: (x+acc, str(x)), 0, range(1,5))
      (10, ['1', '2', '3', '4'])
    """
    acc: T = e
    ys: List[V] = []
    for x in xs:
        acc, c = f(acc, x)
        ys += c
    return (acc, ys)


def mapAccumR(f: Callable[[T, U], Tuple[T, S]], e: T, xs: List[U]) -> Tuple[T, List[S]]:
    """
    mapAccumR transforms the list <xs> with <f> and simultaneously accumulates its elements from right-to-left into a value of <T>.

    Returns:
      Tuple[T, List[S]]:
        A tuple of the accumulated value and the transformed list.

    Examples:
      >>> mapAccumR(lambda acc, x: (x+acc, str(x)), 0, range(1,5))
      (10, ['4', '3', '2', '1'])
    """
    acc: T = e
    ys: List[V] = []
    for x in reverse(xs):
        acc, c = f(acc, x)
        ys += c
    return (acc, ys)


def replicate(n: int, x: T) -> List[T]:
    """
    Replicate the value <x> <n> times.
    When the value of <n> is less than 0, an empty list is returned.

    Returns:
      List[T]:
        The list of length <n> with <x> the value of every element.

    Examples:
      >>> replicate (0, 1)
      []
      >>> replicate (3, "foo")
      ['foo', 'foo', 'foo']
      >>> replicate (-5, 1)
      []
    """
    if n < 0:
        return []

    xs: List[T] = []
    for i in range(n):
        xs.append(x)

    return xs


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
