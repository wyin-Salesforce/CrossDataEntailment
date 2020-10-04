import statistics





# initializing list
test_list = [36.62, 37.16, 38.62, 35.87, 30.87]

print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))

'''
67.93/3.31
'''
