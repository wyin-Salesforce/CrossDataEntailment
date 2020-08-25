import statistics





# initializing list
test_list = [46.23, 40.16, 52.58, 46.84, 60.39]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
