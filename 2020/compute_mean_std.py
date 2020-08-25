import statistics





# initializing list
test_list = [85.02, 83.82, 83.69, 83.96, 83.89]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
