
# coding: utf-8

# In[3]:

import numpy as np


# In[4]:

import matplotlib as mp


# In[5]:

data=np.genfromtxt('minescombineddata.csv', delimiter=',')


# In[8]:

labelY=data[:,19]


# In[9]:

trainX=data[:,0:18]


# In[10]:

s=trainX.shape


# In[11]:

labelY.shape


# In[14]:

theta= np.empty(18) ; print(theta) 


# In[15]:

def newhlog(theta):
    
    h=theta*trainX

    hyp=np.empty(s[0])
    for i in range (0, s[0]-1):
        hyp[i]=np.sum(h[i,:])
 
    hlog=(1/(1+np.exp(-hyp)))
    return hlog


# In[16]:

hlog=newhlog(theta)


# In[17]:

m=s[0] ; lmda=1.5  ;


# In[18]:

labelY


# In[19]:

def cost(hlog_i, label_i) :
    
    if label_i==1:
        return -1*np.log(hlog_i)
    else:
        return -1*np.log(1-hlog_i)


# In[20]:

cost(10**-99,1)


# In[21]:

def J_(theta,hlog, labelY):            #final cost per iteration
    J=0
    for i in range(0,m): 
        J=J+(cost(hlog[i], labelY[i]))/m
                                    
    temp=0
    for j in range(1, 16):
        temp=temp+theta[j]**2                         
    reg_term=(lmda/(2*m))*temp;
    J=J+reg_term;   
    return J


# In[22]:

def dfn_J_j(theta,j,hlog, labelY):             #final differential value of cost wrt a particular weight per iteration
    dfn_J=0 
    for i in range(1, m):
        dfn_J=dfn_J+(hlog[i]-labelY[i])*trainX[i,j] 
    if(j!=0):
        dfn_J=(dfn_J + lmda*theta[j])/m
    else :
        dfn_J=dfn_J/m
    return dfn_J


# In[23]:

def grad_desc(theta,hlog, labelY):          #gradient descent gives new values theta vector
    for j in range(0, 17):
        theta[j]=theta[j]-(.01/m)*dfn_J_j(theta,j,hlog, labelY)   
    hlog=newhlog(theta)


# In[24]:

def train_model(theta,hlog, labelY):
    count=0
    while(count<2500):        
        print( "Iteration {}, Cost = {}".format(count ,J_(theta,hlog, labelY)) )
        grad_desc(theta,hlog, labelY)
        count=count+1


# In[25]:

print("Press y to train:")
ch=input()
if ch=='y':
    train_model(theta,hlog, labelY)


# In[ ]:

print(theta);

