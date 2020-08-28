import statistics





# initializing list
test_list = [84.59, 83.56, 82.22, 84.09, 83.29]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
