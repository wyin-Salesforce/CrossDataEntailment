import statistics





# initializing list
test_list = [95.62, 95.62, 95.67, 95.53, 95.29]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
