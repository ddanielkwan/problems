
c = [0,0,0,1,0,0,1]
current = 0
count = 0
while current < len(c) - 1:
    if current + 2 < len(c) and c[current + 2] ==0 :
        count +=1 
        current +=2
    else:
        count += 1
        current += 1
        
print(f'This many steps {count}')