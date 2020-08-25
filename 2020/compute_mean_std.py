import statistics





# initializing list
test_list = [55.87, 59.40, 52.58, 53.85, 63.73]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
