path = 'UDDDUDUU'
height = 0
valley = 0
for p in path:
    if p == 'U':
        height += 1
        if height == 0:
            valley +=1
    else:
        height -= 1

print(f'This many valleys: {valley}')
