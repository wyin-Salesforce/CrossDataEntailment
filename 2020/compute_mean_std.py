import statistics





# initializing list
test_list = [85.19, 83.52, 83.52, 84.72, 84.66]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
