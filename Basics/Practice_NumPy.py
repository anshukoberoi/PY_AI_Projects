# crud+export
import numpy as np
# C: create
data = np.random.rand(2,3,4)
zeros=np.zeros((2,2,2))
full=np.full((2,2,2),7)
ones=np.ones((2,2,2))
print(data)
print(zeros)
print(full)
print(ones)
arr=np.array([[1,2,3,4],[1,2,3,4]])
print(arr)
print(type(arr))
# R: read
shape = data.shape
size = data.size
types = data.dtype
print(shape,size,types)
data = [[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[13,14,15,16],[17,18,19,20],[21,22,23,24],[25,26,27,28]]]
arr=data[0]
slicer=data[0:2]
reverse = data[-1]
singlevel=data[0][0][0]
print('data:',data,'\narr:',arr,'\nslicer:',slicer,'\nreverse:',reverse,'\nsinglevel',singlevel)
list1=np.random.rand(10)
list2=np.random.rand(10)
add=np.add(list1,list2)
sub=np.subtract(list1,list2)
mul=np.multiply(list1,list2)
div=np.divide(list1,list2)
dot=np.dot(list1,list2)
print('Add:',add,'\nSub:',sub,'\nMul:',mul,'Div:',div,'\ndot:',dot)
# STATISTICAL
sqrt=np.sqrt(25)
ab=np.abs(-2)
power=np.power(2,7)
log=np.log(25)
exp=([2,3])
mins=np.min(list1)
maxs=np.max(list1)
print('sqrt:',sqrt,'ab:',ab,'power:',power,'log:',log,'exp:',exp,'mins:',mins,'maxs:',maxs)
# U: Update
data[0][0][0]=700
print(data)
data.sort()
print(data)
print(data.shape)
data=data.reshape(2,2,-1)
print(data.shape)
zeroes=np.zeros(8)
print(zeroes)
zeroes = np.append(zeroes,[3,4])
print(zeroes)
zeroes=np.insert(zeroes,2,1)
print(zeroes)
print('0data:',data)
np.delete(data,0,axis=1)
print('data:',data)
np.save("new_arr",data)
test=np.load("new_arr.npy")
print('test:',test)