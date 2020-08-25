import statistics





# initializing list
test_list = [85.02, 83.79, 84.19, 84.02, 83.72]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
