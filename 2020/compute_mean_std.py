import statistics





# initializing list
test_list = [50.28, 50.31, 50.05, 50.51, 49.84]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
