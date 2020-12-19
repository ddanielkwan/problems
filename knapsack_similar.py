import math
import random
def maximumToys(prices,budget):
	prices.sort()
	count = 0
	sum = 0

	for i in range(0,len(prices)):
		sum += prices[i]
		if sum > budget:
			count = i 
			break

	return count
if __name__ == "__main__":
	print(maximumToys([1,12,5,111,200,1000,10],50))
