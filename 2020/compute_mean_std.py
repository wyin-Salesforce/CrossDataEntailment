import statistics





# initializing list
test_list = [84.69, 83.92, 83.06, 84.22, 84.19]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
