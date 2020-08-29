import statistics





# initializing list
test_list = [84.69, 84.36, 84.72, 84.46, 84.56]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
