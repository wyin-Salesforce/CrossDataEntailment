import statistics





# initializing list
test_list = [85.62, 84.16, 84.62, 84.16, 83.25]

print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))

'''
67.93/3.31
'''
