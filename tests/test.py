import unittest
from hypothesis import given, strategies as st

from src.basic_iter import prim_list as L


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

    @given(st.lists(st.integers()))
    def test_foldlcons_is_rev(self, xs):
        self.assertEqual(L.reverse(xs), L.foldl(lambda x, acc: [x] + acc, [], xs))

    @given(st.lists(st.integers()))
    def test_append_id(self, xs):
        self.assertEqual(xs, L.append([], xs))
        self.assertEqual(xs, L.append(xs, []))

    @given(st.lists(st.integers()))
    def test_revrev_id(self, xs):
        self.assertEqual(L.reverse(L.reverse(xs)), xs)

    @given(st.lists(st.lists(st.integers())))
    def test_flatten_preserve_len(self, xxs):
        self.assertEqual(sum(L.map(len, xxs)), len(L.flatten(xxs)))

    @given(st.lists(st.lists(st.integers())).filter(lambda x: len(x) >= 1))
    def test_flatten_preserve_element(self, xxs):
        self.assertTrue(L.prefix_of(L.flatten(xxs), xxs[0]))

    @given(st.lists(st.integers()), st.lists(st.integers()))
    def test_prefix_of (self, ls, rs):
        self.assertTrue(L.prefix_of(ls + rs, ls))


if __name__ == "__main__":
    unittest.main()
