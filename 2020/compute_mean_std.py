import statistics





# initializing list
test_list = [84.66, 83.89, 82.12, 84.19, 83.92]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
