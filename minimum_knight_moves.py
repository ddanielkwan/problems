"""
https://www.geeksforgeeks.org/minimum-steps-reach-target-knight/
"""
def minmoves(n, startrow, startcol, endrow, endcol):
    #all possible moves for knight
    x_moves = [2, 2, -2, -2, 1, 1, -1, -1] 
    y_moves = [1, -1, 1, -1, 2, -2, 2, -2] 
    #brute force
    queue = []
    queue.append(cell(startrow, startcol, 0))

    #making the whole board
    visited = [[False for i in range(n + 1)]  
                      for j in range(n + 1)] 
                    
    visited[endrow][endcol] = True
    
    while(len(queue) > 0): 
          
        t = queue[0] 
        queue.pop(0) 
          
        # if current cell is equal to target  
        # cell, return its distance  
        if(t.x == endrow and 
           t.y == endcol): 
            return t.dist       
        # iterate for all reachable states  
        for i in range(8): 
              
            x = t.x + x_moves[i] 
            y = t.y + y_moves[i] 

            queue.append(cell(x, y, t.dist + 1))   
    return -1 
def isInside(x, y, N): 
    if (x >= 1 and x <= N and 
        y >= 1 and y <= N):  
        return True
    return False

class cell: 
      
    def __init__(self, x = 0, y = 0, dist = 0): 
        self.x = x 
        self.y = y 
        self.dist = dist 
        
print(minmoves(8, 0, 0, 0,0))