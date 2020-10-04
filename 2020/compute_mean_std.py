import statistics





# initializing list
test_list = [86.62, 85.37, 86.16, 84.87, 85.62]

print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))

'''
67.93/3.31
'''
