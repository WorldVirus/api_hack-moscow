# coding: utf-8
from magic.magic import MagicWorker

import operator

if __name__ == '__main__':
    magic = MagicWorker()
    s = u'Не знаю, как подлкючиться к интернету'
    print(s)
    ans = magic.predict(s)
    print(ans)
    