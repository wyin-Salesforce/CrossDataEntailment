import statistics





# initializing list
test_list = [65.37, 65.56, 65.75, 66.37, 67.62]

print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))

'''
67.93/3.31
'''
