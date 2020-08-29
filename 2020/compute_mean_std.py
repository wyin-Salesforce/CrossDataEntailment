import statistics





# initializing list
test_list = [84.46, 83.99, 82.09, 85.12, 83.82]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
