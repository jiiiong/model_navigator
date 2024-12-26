from typing import List

import fire


def foo(
    l: List[int],
):
    print(type(l), l)

if __name__ == '__main__':
    fire.Fire(foo)
