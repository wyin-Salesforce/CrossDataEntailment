import statistics





# initializing list
test_list = [80.92, 80.72, 79.75, 79.35, 79.15]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
