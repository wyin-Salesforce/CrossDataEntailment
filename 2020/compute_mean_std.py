import statistics





# initializing list
test_list = [87.81, 86.26, 85.13]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))

'''
2+0.9 source without seed 32
85.08/0.11
'''
