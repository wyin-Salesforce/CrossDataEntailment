import statistics





# initializing list
test_list = [81.98, 81.65, 83.20, 82.59, 81.89]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
