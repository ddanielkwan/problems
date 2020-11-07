arr = [
    [1 ,1 ,1, 0 ,0 ,0],
    [0, 1 ,0 ,0, 0 ,0],
    [1 ,1, 1, 0 ,0 ,0],
    [0 ,0, 2, 4 ,4 ,0],
    [0 ,0 ,0 ,2 ,0 ,0],
    [0 ,0 ,1 ,2 ,4, 0]]

max_sum = 0
for i in range(len(arr) - 2):
    for j in range(len(arr[i])-2):
        h_sum = arr[i][j] + arr[i][j+1] + arr[i][j+2] + arr[i+1][j+1] + arr[i+2][j]+ arr[i+2][j+1] + arr[i+2][j+2]
        max_sum = max(max_sum, h_sum)
print(f'max sum {max_sum}')