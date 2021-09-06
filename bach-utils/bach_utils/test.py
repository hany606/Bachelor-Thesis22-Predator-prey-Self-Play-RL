import bach_utils.list as utlst
import bach_utils.sorting as utsrt

l = ["3", "2", "1"]

print(utlst.get_latest(l))
print(l)
print(utlst.get_first(l))
print(l)
print(utlst.get_sorted(l, utsrt.sort_nicely))
print(l)