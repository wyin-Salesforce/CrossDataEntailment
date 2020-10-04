import statistics





# initializing list
test_list = [13.56, 17.87, 9.12, 6.56, 8.12]

print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))

'''
67.93/3.31
'''
