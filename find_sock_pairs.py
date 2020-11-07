ar = [1,2,1,2,1,3,2, 3]
temp, count = [], 0

for sock in ar:
    if sock in temp:
        count += 1
        temp.remove(sock)
    else:
        temp.append(sock)
print(f'This many pairs of socks: {count}')

