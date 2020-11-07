
s = 'abc'
n = 100
default = s.count('a')
extra_len = n % len(s)
extra_count = s[:extra_len].count('a')

print(f' This many a\'s: {default * (n//len(s)) + extra_count}')