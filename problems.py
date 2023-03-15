#two sums
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
    
        #find the difference
        hashmap = {}
     
            
        for i in range(len(nums)):
            # diff = target - nums[i]
            hashmap[nums[i]] = i
        for i in range(len(nums)):
            diff = target - nums[i]
            # cannot use itself 
            if diff in hashmap and hashmap[diff] != i:
                return [i, hashmap[diff]]
            
#max profit

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        
        buy = 0
        sell = 0
        profit = prices[sell] - prices[buy]
        #[6,2,3,8,10]
        while sell < len(prices): #sell is always moving
            #obviously we want a lower buying price
            if prices[buy] > prices[sell]:
                buy = sell #change buy pointer to where current pointer is
            
            if prices[sell] - prices[buy] > profit: #set new max profit 
                profit = prices[sell] - prices[buy]
            sell += 1 #condition to keep going
        
        if profit >0:
            return profit
        else: 
            return 0


#contains dups
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        
        h = {}
        
        for element in nums:
            if element not in h:
                h[element] = element
            else:
                return True
            
        return False

#product of array except itself
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
       
    
        r = [1] * len(nums)
        #prefix first
        prefix = 1 
        for index in range(len(nums)):
            r[index] = prefix
            prefix *= nums[index]
            
        #postfix
        postfix = 1
        for index in range(len(nums)-1, -1, -1):
            r[index] *=  postfix
            postfix *= nums[index]
            
        return r
            
#max contiguous subarray

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
       
        s = 0 
        m = nums[0]
            
        #check if sum is negative reset
        for ele in nums:
            if s < 0:
                s = 0
            s += ele
            m = max(m, s)
            
        return m 
            

#find min of sorted rotated
class Solution:
    def findMin(self, nums: List[int]) -> int:
        
        
        mid = len(nums) //2 
        left = 0
        right = len(nums)-1
        
        m = nums[0]
      
        
        while left <= right:
            #if sorted
            if nums[left] < nums[right]:
                m = min(m, nums[left])
                break
            
            #[3,4,5,1,2]
            mid = (left + right)// 2
            m = min(m, nums[mid])
            
            if nums[mid] >= nums[left]: #search right, because it means
                #mid is on the left ascending sorted
                left = mid + 1
            else:
                right = mid - 1
                
        
        return m

#find index of sorted roatted array

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) -1
        
        res = -1
                
        while left <= right:
            mid = (left + right)//2
            if target == nums[mid]:
                return mid
            # 4 for [3,4,5,1,2]
            if nums[left] <= nums[mid]: #perfect ascending
                if target > nums[mid]:
                    left = mid + 1
                elif target < nums[left]:
                    left = mid + 1
                else: #since we already fchecked for middle
                    right = mid - 1
            
            #[5,1,2,3,4]
            else: #nums[left] > nums[mid]
                if target < nums[mid]:
                    right = mid - 1
                elif target > nums[right]:
                    right = mid -1
                else:
                    left = mid + 1
                    
        return res


#three sum 
#O(nlogn) due to sort + O(n^2)
class Solution:
    #Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that 
    # i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        
        res = []
        
        
        nums.sort()
        
        for i, a in enumerate(nums):
            if i > 0 and a == nums[i-1]: 
                #if its not the first element and its not hte same as previosu
                
                continue
                
            l, r = i+1 , len(nums) - 1

            while l < r:
                threeSums = a + nums[l] + nums[r]
                if threeSums > 0:
                    r = r - 1
                elif threeSums < 0:
                    l = l + 1
                else:
                    res.append([a,nums[l], nums[r]])
                    l += 1
                    #and if you dont want to have the same sum
                    while nums[l] == nums[l-1] and l < r:
                        l += 1
                            
        return res
            
        

#max Container O(n) area
class Solution:
    def maxArea(self, height: List[int]) -> int:
        
        
    
        area = 0
        left, right = 0, len(height) - 1
        while left < right:
            
            
            
            area = max(area, min(height[left], height[right]) * (right-left))
            
            if height[left] > height[right]:
                right -= 1
            elif height[right] > height[left]:
                left += 1
            else: #equal
                right -= 1

            
        return area



#climbing stairs fibonacci
class Solution:
#     You are climbing a staircase. It takes n steps to reach the top.

# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
    def climbStairs(self, n: int) -> int:
        
        
        one,two = 1,1 
        
        for i in range(n-1):
            #5, 8 
            #this is just fibonaci
            #add the previous two
            temp = one
            one = one + two
            two = temp
            
        return one 



#changing coin O(n2)
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        
      
        dp = [amount + 1] * (amount + 1)
        #b/c going 0..amount
        #^ amount +1 is jsut max value or can math.maxvalue

        dp[0] = 0
        #base case

        #bottom up
        for amoun in range(1,amount+1):

            for coin in coins:
                if amoun - coin >= 0:
                    #1 because we use one coin from the for loop
                    dp[amoun] = min(dp[amoun], 1 + dp[amoun-coin])
                    #because ex: on coin 4
                    #amount = 7
                    #dp[7] = 1 + dp[amoun-c]

        return dp[amount] if dp[amount] != amount +1 else -1
    #can only return if its found else its -1



#longest increaseing subsequ
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        LIS = [1] * len(nums)
        
        
        #iterate frm every index frm range in reverse
        for i in range(len(nums), -1,-1):
            for j in range(i+1, len(nums)):
                if nums[i] < nums[j]:
                    LIS[i] = max(LIS[i], 1+  LIS[j])
                    
        return max(LIS)


#longest common subsequence
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        #O(n*m) and memory too
        dp = [[0 for j in range(len(text2) + 1)] for value in range(len(text1)+1)]
        
        for i in range(len(text1)-1,-1,-1):
            for j in range(len(text2)-1,-1,-1):
                if text1[i] == text2[j]:#if match
                    dp[i][j] = 1 + dp[i+1][j+1] # add to the diagonal
                else:#dont match
                    #max of two values, to the right and bottom
                    dp[i][j] = max(dp[i][j+1], dp[i+1][j])
                    
        return dp[0][0]


#word break
#O(n*m)
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
        dp = [False] * (len(s)+1)
        #+1 for base base
        dp[-1] = True
        
        for i in range(len(s)-1,-1,-1):
            #go frm bottom up
            #try every signle word from dict and see if it matches
            for word in wordDict:
                if (i+len(word)) <= len(s) and s[i:i+len(word)]==word:#there are enoguh characters      
                    dp[i] = dp[i + len(word)]
                    
                #now if there is a word that is segmented in the dictionary
                #we want to break out of the current index
                if dp[i] is True:
                    break
        return dp[0]
                    

#combination sum O(n) * O(m)
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = {0: 1}
        
        for total in range(1, target+1 ):
            dp[total] = 0#just init
            for num in nums:
                dp[total] += dp.get(total-num,0) #if exist get 0 else
                
                
                
        return dp[target]


#house robber ii
#O(n) O(1) memory
class Solution:
    def rob(self, nums: List[int]) -> int:
        
        return max(nums[0], self.helper(nums[1:]), self.helper(nums[:-1]))
    
    
    def helper(self, nums):
        
        rob1, rob2 = 0, 0
        #refer to rob 1
        
        
        for num in nums:
            temp = max(rob1 +num, rob2)
            rob1 = rob2
            rob2 = temp
            
        return rob2


#unique paths O(n)

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        row = [1] * n
        
        for i in range(m-1):
            newRow = [1] * n
            for j in range(n-2,-1,-1):
                
                newRow[j] = newRow[j+1] + row[j]
                
            row = newRow
            
        return row[0]

#possible jump 
class Solution:

    def canJump(self, nums: List[int]) -> bool:
        goal = len(nums) - 1
        
        
        for i in range(len(nums)-1, -1,-1):
            if i +nums[i] >= goal:
                goal = i

#inplace replace matrix zero
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        
        #note we are not looping for first col and row because using it
        is_col = False
        R=len(matrix)
        C=len(matrix[0])
        
        for i in range(R):
            #since first cell for both row and first col is same
            #matrix[0][0]
            #can use additional variable for either first row.col
            #for this we use col varialbe
            if matrix[i][0] == 0:
                is_col = True
            for j in range(1,C):
                #if element is zero set first elemnt of row and colm to 0
                if matrix[i][j]==0:
                    matrix[0][j] = 0
                    matrix[i][0]= 0
                    
        #do it again and use flag to update
        for i in range(1, R):
            for j in range(1,C):
                if matrix[i][0] == 0 or matrix[0][j]==0:
                    matrix[i][j]=0
        
        #check if first row and col need to set zero
        if matrix[0][0] == 0:
            for j in range(C):
                matrix[0][j]=0
                
        if is_col:
            for i in range(R):
                matrix[i][0]=0
         
        
        
        
        
        return True if goal == 0 else False


#reverse linked ist
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        prev, curr = None, head
        
        while curr:
            temp = curr.next
            curr.next = prev
            
            prev = curr
            curr =  temp #to keep going but reember we set curr.next to prev so we need temp vairable
            
            
        return prev
            
            
#character repalcement
# O(26N) hashmap     
  class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        
#         maxValueToReturn, runningCount = 0, 0
#         temp_k = k
#         currentChar = s[0] 
#         for stringIndex in range(0, len(s)):
            
#             if s[stringIndex] == currentChar:
#                 runningCount += 1
#                 maxValueToReturn = max(runningCount, maxValueToReturn)
#             elif s[stringIndex] != currentChar:
#                 if temp_k > 0:
#                     temp_k -= 1
#                     runningCount += 1
#                     maxValueToReturn = max(runningCount, maxValueToReturn)
#                 else:
#                     currentChar = s[stringIndex]
#                     maxValueToReturn = max(runningCount, maxValueToReturn)
#                     runningCount = 1
#                     temp_k = k
                    
     
            
        
        
#         return maxValueToReturn
        count = {}
        res = 0
        
        l = 0
        for r in range(len(s)):
            count[s[r]] = 1 + count.get(s[r],0)
            
            if (r-l+1) - max(count.values()) > k:
                count[s[l]] -=1
                l +=1
            res=max(res,r-l+1)
        return res

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        
        if not root:
            return 0
        
        else:
            return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
#DFS
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        
        if not root:
            return None
        
      
        temp = root.left

        root.left = root.right
        root.right = temp

        self.invertTree(root.left)
        self.invertTree(root.right)
        return root

class Solution:
    def longestPalindrome(self, s: str) -> str:
        
        res = ""
        reslen = 0
        for i in range(len(s)):
            #odd
            l,r = i,i 
            while l >= 0 and r < len(s) and s[l]==s[r]:
                if (r-l+1) > reslen:
                    res = s[l:r+1]
                    reslen = len(res)
                l-=1
                r +=1
            
            #even case
            l,r = i, i+1
            while l>=0 and r<len(s) and s[l]==s[r]:
                if r-l+1 > reslen:
                    res = s[l:r+1]
                    reslen = len(res)
                l-=1
                r +=1
                
            
        return res
                    

#O(n) copy graph
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        hashmap = {}
        
        def dfs(node):
            
            if node in hashmap:
                return hashmap[node]
            
            if node == None:
                return None
            copyNode = Node(node.val)
            hashmap[node] = copyNode
            for neighbour in node.neighbors:
                copyNode.neighbors.append(dfs(neighbour))
                
            return copyNode
        
        return dfs(node)


## Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        
        dummy = ListNode()
        tail = dummy
        
        while l1 and l2:
            
            if l1.val < l2.val:
                tail.next = l1
                l1 = l1.next
            
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        
        if l1:
            tail.next = l1
        elif l2:
            tail.next = l2
        return dummy.next

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
#         nums = sorted(list(set((nums))))
        
        
        maxReturn = 0
        
#         tempcounter = 1
   
#         for index in range(len(nums)):
            
            
            
#             if (index+1 < len(nums)) and (nums[index] + 1) == nums[index+1] :
                
#                 tempcounter += 1
            
#             else:
#                 tempcounter = 1
            
#             maxReturn = max(tempcounter, maxReturn)
        
#         return maxReturn
        nums = set(nums)
        
        for n in nums:
            
            if (n-1) not in nums:
                temp = 1
                #means its a start
                while n +1 in nums:
                    temp +=1
                    n += 1
            else:
                temp = 1
            
            maxReturn = max(temp, maxReturn)
            
        return maxReturn
            
            
            
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


#1 Yes that symmetric
#1 2 Not symmetric
#1 2 2 Symmetric
#1 2 3 Not symmetric


#input as bfs
#[1, 2^0
# 2,2, 2^1
# 3,4,4,3, 2^2
# 5,6,5,6,6,5,6,5] 2^3
#every symmetric tree has to be odd

#
class Solution:
    def isMirror(self, root1, root2):
        #condition to stop recursion
        # print(root)
        if root1 == None and root2 == None:
            return True
        if root1 == None or root2 == None:
            return False


        return ((root1.val == root2.val) and (self.isMirror(root1.left, root2.right) and(self.isMirror(root2.right, root1.left))))

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        return self.isMirror(root,root)
    
    
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        
        i = 0
        j = 0

        while i < len(s) and j < len(t):

            if s[i]== t[j]:
                i +=1
                j += 1
            else:
                j += 1

        if i == len(s):
            return True

        return False
class Solution:
    def isPalindrome(self, x: int) -> bool:
        


        if x < 0 :
            return False
    
        x = str(x)

        l = 0
        r = len(x) - 1
        while l < r:

            if x[l] != x[r]:
                return False
            
            l +=1 
            r -= 1



        return True

class Solution:
    def romanToInt(self, s: str) -> int:

        temp = 0
        prev = None
        roman = {}
        roman["I"]  =       1
        roman["V"]         =    5
        roman["X"]       =      10
        roman["L"]       =      50
        roman["C"]        =     100
        roman["D"]         =    500
        roman["M"]          =   1000
        for index in range(len(s)):
            #checking in front because special case 
            #normally it goes big to smallest
            #if it is smaller that means we subtract
            if index + 1 < len(s) and roman[s[index]] < roman[s[index+1]]:
                    temp -= roman[s[index]]
            else:
                temp += roman[s[index]]

        return temp

            

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    #O(p+q)
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        #Would assume 


        if p == None and q == None:
            return True
 
        if p == None or q == None or p.val != q.val:
            return False
     
      
        
        return (self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right))
        #base case
        

    
    def isSame(self,p, q):
        

        if p == None and q == None:
            return True
        
        if p == None or q == None:
            return False
        if p.val != q.val:
            return False
        
        
        else:
            
            return self.isSame(p.left, q.left)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        #O(n/2) max levels
        #O(n)

     
     #   3
     #  9 20
     #     15 7

     #[3, [9,20], [null, null, 15,7]]

        queue = [root]
        ret = []
        

        while queue:
            level = []
            #append current node to queue FIFO
            #appending values
            qLen = len(queue)

            for i in range(qLen):
                x = queue.pop(0)
                if x:
                    level.append(x.val)

                    queue.append(x.left)
                    queue.append(x.right)

            if level:
                ret.append(level)
           

        return ret



class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
       
        # if nums == []:
        #     return -1

            
        #     # for index in range(len(nums)):
        #     #     if nums[index] == target:
        # #         return index
        # found = -1
        # index = 0
        # while index < len(nums) and nums[index] <= target :
            
        #     if nums[index] == target:
        #         found = 1
        #         return index
        #     else:# not equal 
        #         #return the index that it would be
            
        #         index +=1 
    

        # return index 

        #ologn
        l,r = 0, len(nums) - 1
        while l <= r:

            mid = (l + r )//2
            if nums[mid] == target:
                return mid
            elif target  > nums[mid]:
                l += 1
            else: #target < nums[mid]
                r -= 1
            
        return l

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:

    def __init__(self):
        self.ans = 0
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        self.dfs(root, 0)

        return self.ans
    

    def dfs(self, node, value):

        if not node:
            return
        
        value *= 10
        value += node.val

        if node.left == None and node.right == None:
            self.ans += value

        self.dfs(node.left, value)
        self.dfs(node.right, value)

        return

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # abcabcabcd 4
        # a 1
        # ab 2
        # abc 3
        # aa 1
        maxLength = 0
        hashmap = {} #keep track of the index of character
        substring = ""
        left = 0
        right = 0
        #acabcbb
        #pwwkew
        for right in range(len(s)): #loop through the string sequence
            
            while s[right] in hashmap:
                hashmap.pop(s[left])
                left += 1
                

            hashmap[s[right]] = right

            
            maxLength = max(maxLength, right-left + 1)
        return maxLength







class Solution:
    def addBinary(self, a: str, b: str) -> str:

        result = ""
        carry = 0

        #we should reverse the string because
        #in binary its big number first
        #in addition we add last digit right to left firsrt

        a = a[::-1]
        b = b[::-1]

        #loop through the max length of the two
        for i in range(max(len(a), len(b))):
            #but we want interger not character
            digitA = ord(a[i]) - ord("0") if i <len(a) else 0
            digitB = ord(b[i]) - ord("0") if i < len(b) else 0

            total = digitA + digitB + carry
            char = str(total % 2) #because 1 + 1 = 2 and theres 1 carry 2 %2 = 1
            result = char + result
            carry = total // 2
        #we find if theres a carry
        if carry:
            result = "1" + result
        return result

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:

        if root is None:
            return 0
    
        left = self.minDepth(root.left) 
        right = self.minDepth(root.right) 

        if (left == 0 or right == 0): #skew tree
            return 1+ max(left, right)
        return 1 + min(left, right) 
    
 

       class Solution:
    def countBits(self, n: int) -> List[int]:

        #refer to notes
        offset = 1 
        dp = [0] * (n + 1 )



        for i in range(1, n + 1):
            if offset * 2 == i:
                offset = i
            
            dp[i] = 1+  dp[i- offset]

        return dp

class Solution:
    def pivotInteger(self, n: int) -> int:

        
  
        #n(n-1) /2 

        for i in range(1, n + 1):
            left = (i*(i+1))/2
            right = (n*(n+1))/2 - (i*(i-1))/2

            if right == left:
                return i

        return -1

        

class Solution:
    def getMaximumGenerated(self, n: int) -> int:

        
        arr = list(range(0,n+1))



        for i in arr:
            if i % 2 == 0:
                arr[i] = arr[i//2]
            else:
                arr[i] = arr[i//2] + arr[i//2 + 1]

        return max(arr)
           

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """

        last = m + n - 1
        while m > 0 and n > 0:
            #while elements left in both arrays
            if nums1[m-1] > nums2[n-1]:
                nums1[last] = nums1[m-1]
                m -= 1
            else:
                nums1[last] = nums2[n-1]
                n -= 1
            last -= 1
        #edge case what if 
        # [2,2,3,0,0,0]
        # [1,2,3]
        #what if the second is already sorted in order so we jsut 
        #fill the remaining

        while n > 0:
            nums1[last] = nums2[n-1]
            n -= 1
            last -=1
            
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        #use the zeros in the beginning and end
        #so we cna add

    #O(n^2)
        result = [[1]]

        for i in range(numRows-1):
            #number of elements in each row is the one before + 1
            temp = [0] + result[-1] + [0]
            row = []
            for j in range(len(result[-1]) + 1):
                row.append(temp[j] + temp[j+1])
            
            result.append(row)

        return result

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:

        def dfs(root) -> list:

            if not root:
                return [True, 0]
            
            left, right = dfs(root.left), dfs(root.right)
            balanced = (left[0] and right[0] and (abs(left[1]- right[1]) <= 1))

            return [balanced, 1 + max(left[1], right[1])]
        
        return dfs(root)[0]


class Solution:
    #o(n)
    def decodeMessage(self, key: str, message: str) -> str:
        str1 = ""
        res = ""
        for i in key:
            if i != " " and i not in str1:
                str1+=i

        for i in message:
            if i != " ":
                res+= chr(str1.index(i)+97)
            else:
                res+=" "
        return res
