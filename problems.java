class Solution {
    public int searchInsert(int[] nums, int target) {
        


        int l = 0;
        int r =  nums.length;


        while (l< r){
            int mid = (l + r)/2;

            if (nums[mid] == target){
                return mid;
            }
            else if (target > nums[mid]){
                l +=1;
            }
            else{
                r -= 1;
            }
        }
        return l ;

    }
}


/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
 // 
 class Solution {

    //going through each node and adding that node to the current sum and checking if it is equal to the target
    //keep going in pre order
    public boolean hasPathSum(TreeNode root, int targetSum) {


            if (root == null){
                return false;
            }
       
            // if we reach the end and it is a leaf node and its
            // value is target
            if (root.left == null && root.right == null){
                if (root.val==targetSum){
                    return true;
                }
            }
          
            return (hasPathSum(root.left, targetSum-root.val) || hasPathSum(root.right,targetSum-root.val));
            
            
    }

    
}

//binary search
class Solution {
    public int search(int[] nums, int target) {

        // left and right and mid point
        int left = 0;
        int right = nums.length - 1;

        // make sure the left doesnt exceed the right
        // -1, 0 ,3, 5 ,9 , 12
        while (left <= right){

            int middleIndex = right - left ;
            int valueAtMiddle = nums[middleIndex];

            if (valueAtMiddle == target){
                return middleIndex;
            } 
            else{
                //not middle value
                if (target > valueAtMiddle){
                    left = middleIndex + 1;
                }
                else if (target < valueAtMiddle){
                    right = middleIndex - 1;
                }
            }

            
        }
        return -1;

        
    }
}

class Solution {
    public String longestCommonPrefix(String[] strs) {
        //O(n) number of characters in the first string
        StringBuilder result = new StringBuilder("");
        //we can append

        for (int index = 0; index < strs[0].length(); index++){
            
            for (String s:strs){
                if (index == s.length() || s.charAt(index) != strs[0].charAt(index)){
                    return result.toString();
                }
                
            
            }
            result.append(strs[0].charAt(index));
        }
        return result.toString();
    }
}

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int ans = 0;
    public int sumNumbers(TreeNode root) {

        if (root == null){
            return 0;
        }
        dfs(root, 0);
        return ans;
       
    }

    public void dfs(TreeNode node, int value){

        if (node == null){
            return;
        }
        // if not null
        value *= 10 ;//left shift
        value += node.val;
        
        if (node.left == null && node.right == null){
            ans += value;
        }


        dfs(node.left, value);
        dfs(node.right, value);

    }
}

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        

        ListNode current = head;

        while (current != null){ // start looping until the end

            ListNode temp = current;

            while (temp != null && temp.val == current.val){
                temp = temp.next;
            }

            current.next = temp;
            current = current.next;

        
        
        }
            
  return head;
            
        }
             
    
}