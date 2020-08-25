import statistics





# initializing list
test_list = [81.56, 81.84, 81.56, 81.51, 81.74]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
