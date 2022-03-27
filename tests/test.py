import importlib

import unittest
import doctest
from hypothesis import assume, given, example, strategies as st

import src.basic_iter
from src.basic_iter import prim_list as L

import pkgutil


def test_targets(pkg) -> pkgutil.ModuleInfo:
    """
    Enumerate modules under `pkg` recursively.
    """
    return pkgutil.walk_packages(path=pkg.__path__, prefix=pkg.__name__ + ".")


def load_tests(loader, tests, ignore):
    """
    Scan and register all test cases defined as doctests in the `src.basic_iter` module.
    """
    for (_, name, _) in test_targets(src.basic_iter):
        print(f"loading doctests in {name}...")
        importlib.import_module(name)
        tests.addTests(doctest.DocTestSuite(name))
    return tests


class TestPrimList(unittest.TestCase):
    def test_find(self):
        self.assertEqual(1, L.find(1, [1, 2, 3]))
        self.assertIsInstance(L.find(1, [2, 3, 4]), L.NotFound)

    def test_find_if(self):
        self.assertEqual(2, L.find_if(lambda x: x % 2 == 0, [1, 2, 3]))
        self.assertIsInstance(L.find(1, [2, 3, 4]), L.NotFound)

    def test_append(self):
        self.assertEqual([4, 5, 6, 1, 2, 3], L.append([4, 5, 6], [1, 2, 3]))
        self.assertEqual("helloworld", L.append("hello", "world"))
        self.assertEqual([(1, "foo"), (2, "bar")], L.append([(1, "foo")], [(2, "bar")]))

    @given(st.lists(st.integers()), st.lists(st.integers()))
    def test_append_iso_len(self, xs, ys):
        self.assertEqual(len(xs) + len(ys), len(L.append(xs, ys)))

    @given(st.lists(st.integers()))
    def test_foldlcons_is_rev(self, xs):
        self.assertEqual(L.reverse(xs), L.foldl(lambda x, acc: [x] + acc, [], xs))

    @given(st.lists(st.integers()))
    @example([])
    def test_append_id(self, xs):
        self.assertEqual(xs, L.append([], xs))
        self.assertEqual(xs, L.append(xs, []))

    @given(st.lists(st.integers()))
    @example([])
    def test_revrev_id(self, xs):
        self.assertEqual(L.reverse(L.reverse(xs)), xs)

    @given(st.lists(st.lists(st.integers())))
    @example([])
    @example([[]])
    def test_concat_preserve_len(self, xxs):
        self.assertEqual(sum(L.map(len, xxs)), len(L.concat(xxs)))

    @given(st.lists(st.lists(st.integers())).filter(lambda x: len(x) >= 1))
    def test_concat_preserve_element(self, xxs):
        self.assertTrue(L.is_prefix(xxs[0], L.concat(xxs)))

    @given(st.lists(st.integers()))
    @example([])
    def test_concat_map(self, xs):
        self.assertEqual(len(xs) * 2, len(L.concat_map(lambda x: [x, x+1], xs)))

    @given(st.lists(st.integers()), st.lists(st.integers()))
    def test_is_prefix(self, ls, rs):
        self.assertTrue(L.is_prefix(ls, ls + rs))

    def test_group_by_empty_id(self):
        self.assertEqual([], L.group_by(lambda x, y: True, []))

    @given(st.lists(st.integers()))
    def test_group_by_true_id(self, xs):
        assume(xs)
        self.assertEqual([xs], L.group_by(lambda x, y: True, xs))

    def test_group_empty_id(self):
        self.assertEqual([], L.group([]))

    @given(st.lists(st.integers()))
    @example([])
    def test_group_preserve_len(self, xs):
        self.assertEqual(len(xs), sum(L.map(len, L.group(xs))))

    @given(st.lists(st.integers()))
    @example([])
    def test_group_specific_group_by(self, xs):
        self.assertEqual(L.group_by(lambda x, y: x == y, xs), L.group(xs))

    @given(st.lists(st.integers()), st.lists(st.lists(st.integers())))
    @example([], [])
    @example([], [[]])
    def test_intercalate(self, xs, xxs):
        self.assertEqual(L.intercalate(xs, xxs), L.concat(L.intersperse(xs, xxs)))

    @given(st.lists(st.lists(st.integers())))
    @example([])
    @example([[]])
    @example([[], []])
    def test_transpose(self, xxs):
        self.assertEqual(sum(map(len, L.transpose(xxs))), sum(map(len, xxs)))

    @given(st.lists(st.integers(), max_size = 13))
    @example([])
    def test_subsequences(self, xs):
        def subseq(ys, zs):
            if [] == ys:
                return True
            if [] == zs:
                return False
            if ys[0] == zs[0]:
                return subseq(ys[1:], zs[1:])
            else:
                return subseq(ys, zs[1:])
        self.assertTrue(L.all(lambda ss: subseq(ss, xs), L.subsequences(xs)))


if __name__ == "__main__":
    unittest.main()
