import statistics





# initializing list
test_list = [95.15, 94.92, 94.73, 95.24, 95.20]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
