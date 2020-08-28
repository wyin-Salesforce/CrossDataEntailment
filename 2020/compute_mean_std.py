import statistics





# initializing list
test_list = [83.39, 81.36, 78.99, 81.72, 83.99]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
