# Example of kNN implemented from Scratch in Python exemplo retirado de
#http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
import knn

testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b'],[4,5,5,'b']]
predictions = ['a', 'a', 'a', 'a', 'b', 'b', 'b','b','a']
accuracy = knn.getAccuracy(testSet, predictions)
print(accuracy)

