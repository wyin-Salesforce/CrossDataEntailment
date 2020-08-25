import statistics





# initializing list
test_list = [84.02, 84.22, 83.49, 83.79, 83.79]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
