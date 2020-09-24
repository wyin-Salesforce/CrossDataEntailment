import statistics





# initializing list
test_list = [45.68, 57.13, 42.15]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))

'''
2+0.9 source without seed 32
85.08/0.11
'''
