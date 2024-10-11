import numpy as np

from itertools import zip_longest

li = [['Gauri', 'Riya'], ['Shubham', 'Ram'], ['Mahi', 'Ravi'], ['are good']]
ans = [''.join(i) for i in zip_longest(*li, fillvalue='')]
print(str(ans))

mlen = max(len(sl) for sl in li)
tlist = [sl+['']*(mlen-len(sl)) for sl in li]
arr = np.array(tlist)
tarr = arr.T
ans = [''.join(i) for i in tarr]
print(str(ans))
