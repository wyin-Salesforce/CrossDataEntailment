import statistics





# initializing list
test_list = [86.26, 86.19, 86.69, 86.02, 86.16]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
