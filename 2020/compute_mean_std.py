import statistics





# initializing list
test_list = [82.03, 82.64, 83.25, 81.56, 81.84]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
