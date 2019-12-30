#!/usr/bin/env python
# coding: utf-8

# # Largest and smalist

# In[ ]:


largest = None
smallest = None
num = input("Enter a number: ")
largest =int(num)
smallest = int(num)
while True:
    num = input("Enter a number: ")
    if num == "done" : break
    try:
        num=int(num)
        if(largest<num):
            largest=num
        if(smallest>num):
            smallest=num
    except:
        print("Invalid input")

print("Maximum is", largest)
print("Minimum is", smallest)


# # Transpose and Flatten

# In[ ]:


import numpy

N,M = list(map(int,input().split()))
x=list()
for i in range(N):
    x.append(list(map(int,input().split())))
    
my_array = numpy.array(x)

print (numpy.transpose(my_array))
print (my_array.flatten())


# # concatenate

# In[ ]:


import numpy

N,M,P = list(map(int,input().split()))

x=list()
y=list()

for i in range(N):
    x.append(list(map(int,input().split())))

for i in range(M):
    y.append(list(map(int,input().split())))

array_1 = numpy.array(x)
array_2 = numpy.array(y)
  

print(numpy.concatenate((array_1, array_2)))   


# In[ ]:


import numpy

array_1 = numpy.array([[1,2,3],[0,0,0]])
array_2 = numpy.array([[0,0,0],[7,8,9]])

print (numpy.concatenate((array_1, array_2), axis = 0))   


# In[ ]:


import numpy as np
N,M,R =3,3,3# list(map(int,input().split()))
zeros=np.zeros((N,M),dtype=np.int)
ones=np.ones((N,M),dtype=np.int)
x=[]
print('[',end='')
for i in range(R):
    if i==(R-1):
        print(zeros,end='')
    else:
        print(zeros,"\n")

print(']')
print('[',end='')
for i in range(R):
    if i==(R-1):
        print(ones,end='')
    else:
        print(ones,"\n")
    
print(']',end='')


# # List Comprehensions:
# Print the list in lexicographic increasing order.

# In[ ]:


x = int (input())
y = int (input())
z = int (input())

n = int (input())
print ([ [ i, j,k] for i in range( x + 1) for j in range( y + 1) for k in range( z + 1) if ( ( i + j +k) != n )] )


# In[28]:


from scipy import signal, misc
import matplotlib.pyplot as plt
import numpy as np

image = misc.face(gray=True).astype(np.float32)

derfilt = np.array([1, -2,1], dtype=np.float32)
ck = signal.cspline2d(image, 8)
deriv = (signal.sepfir2d(ck, derfilt, [1]) + signal.sepfir2d(ck, [1], derfilt))

plt.figure()
plt.imshow(image)
plt.show()
plt.imshow(deriv )
plt.gray()
plt.title('Original image')
plt.show()


# # Maximum without repeting

# In[13]:


n = int(input())
x = list(map(int, input().split()))


maxx=x[0]
for i in x:
    if maxx<i:
        maxx=i
for i in range(n):
    try:
        x.remove(maxx)
    except:
        pass
maxx=x[0]
for i in x:
    if maxx<i:
        maxx=i
print(maxx)


# # Sort Students

# In[6]:



N = int(input())

students = list()
for i in range(N):
    students.append([input(), float(input())])

scores = list(set([students[x][1] for x in range(N)]))
scores.sort()

students = [x[0] for x in students if x[1] == scores[1]]
students.sort()

for s in students:
    print (s)


# # Avarage of selected student

# In[3]:


n = int(input())
student_marks = {}
for _ in range(n):
    name, *line = input().split() # *line takes all the residual values after name
    scores = list(map(float, line))
    student_marks[name] = scores
query_name = input()
sum=0
for i in student_marks[query_name]:
    sum+=i
print("{0:.2f}".format(sum/3))


# # IterTools
# permutations

# In[12]:


from itertools import permutations
string,n = input().split()
for premutate in sorted(list(permutations(string,int(n)))):
    for i in premutate:
        print(i,end='')
    print('')


# combinations

# In[34]:


from itertools import combinations
string,n = input().split()

for i in range(1, int(n)+1):
    for j in combinations(sorted(string),i): #sort the string before enter the function
        print (''.join(j))


# combinations_with_replacement

# In[35]:


from itertools import combinations_with_replacement
string,n = input().split()

for j in combinations_with_replacement(sorted(string),2): #sort the string before enter the function
    print (''.join(j))


# In[ ]:




