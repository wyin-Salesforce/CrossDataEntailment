import statistics





# initializing list
test_list = [42.94, 39.60, 39.18, 49.95, 60.39]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
