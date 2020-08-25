import statistics





# initializing list
test_list = [82.17, 81.60, 81.75, 81.51, 81.60]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
