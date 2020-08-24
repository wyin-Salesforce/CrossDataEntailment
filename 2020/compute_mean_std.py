import statistics





# initializing list
test_list = [50.71, 50.28, 50.31, 50.71, 49.71]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
