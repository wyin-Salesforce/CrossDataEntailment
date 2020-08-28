import statistics





# initializing list
test_list = [84.62, 84.49, 82.72, 83.86, 83.99]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
