import statistics





# initializing list
test_list = [84.76, 84.16, 81.52, 83.96, 84.16]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
