import statistics





# initializing list
test_list = [50.21, 50.65, 49.61, 50.85, 49.81]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
