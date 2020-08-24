import statistics





# initializing list
test_list = [50.35, 51.11, 49.01, 51.98, 50.21]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
