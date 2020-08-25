import statistics





# initializing list
test_list = [95.06, 95.11, 94.96, 94.63, 94.96]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
