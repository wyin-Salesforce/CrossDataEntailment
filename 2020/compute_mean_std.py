import statistics





# initializing list
test_list = [85.09, 83.76, 83.82, 84.22, 83.49]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
