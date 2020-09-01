import statistics





# initializing list
test_list = [84.56, 85.26, 84.16, 84.82, 85.26]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
