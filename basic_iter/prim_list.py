

class NotFound:
    def __init__ (self, cond):
        self.cond = cond

    def __bool__ (self):
        return False

    def __str__ (self):
        if callable(self.cond):
            return f"Not found for the condition {self.cond}"
        else:
            return f"Not found for the value {self.cond}"


def find (e, xs):
    if e in xs:
        return e
    return NotFound(e)


def find_if (p, xs):
    for x in xs:
        if p(x):
            return x
    return NotFound(p)


def append (xs, ys):
    return xs + ys


def map (f, xs):
    return [ f(x) for x in xs ]


def reverse (xs):
    return xs[-1:-len(xs)-1:-1]


def foldl (f, e, xs):
    acc = e
    for x in xs:
        acc = f (x, acc)
    return acc


def scanl (f, e, xs):
    acc = e
    res = [acc]
    for x in xs:
        acc = f (x, acc)
        res = [acc] + res
    return reverse(res)


def foldr (f, e, xs):
    acc = e
    for x in reverse(xs):
        acc = f (x, acc)
    return acc


def scanr (f, e, xs):
    acc = e
    res = [acc]
    for x in reverse(xs):
        acc = f (x, acc)
        res = [acc] + res
    return res


def zipWith (f, xs, ys):
    assert len(xs) == len(ys), "required to be the same length"
    itx = xs
    ity = ys
    res = []
    while itx:
        x = itx[0]
        y = ity[0]
        itx = itx[1:]
        ity = ity[1:]
        res = [f(x, y)] + res
    res.reverse()
    return res


def zip (xs, ys):
    return zipWith(lambda x,y: (x,y), xs, ys)


def unfoldr (f, init):
    res = []
    elm = init
    while True:
        r = f (elm)
        if r:
            (x, e) = r
            res = [x] + res
            elm = e
        else:
            break
    res.reverse()
    return res


