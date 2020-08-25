import statistics





# initializing list
test_list = [48.25, 39.74, 39.41, 52.91, 60.39]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
