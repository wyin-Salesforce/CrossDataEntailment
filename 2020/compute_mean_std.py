import statistics





# initializing list
test_list = [59.37, 60.75, 57.87, 59.37, 58.75]

print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))

'''
67.93/3.31
'''
