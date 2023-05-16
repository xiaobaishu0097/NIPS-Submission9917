import numpy as np
import math
import random
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--n_nodes', type=int, default=200,)
parser.add_argument('--n_timepoints', type=int, default=200,)
args = parser.parse_args()

random.seed(12)
T=args.n_timepoints
N=args.n_nodes

A=np.zeros((N,N))
for i in range(0,N-1,1):
    for j in range(i+1,N,1):
        probtemp=np.random.multinomial(1,pvals=[30/N,0.5*N**(-0.7),0.5*N**(-0.7),1-30/N-N**(-0.7)],size=1)
        if probtemp[0][0]==1:
            A[i,j]=A[j,i]=1
        else:
            if probtemp[0][1]==1:
                A[i,j]=1
            else:
                if probtemp[0][2]==1:
                    A[j,i]=1
ni=sum(A.T)
W=np.dot(np.diag(1/ni),A)
beta=np.array([0.2,0.3,-0.1])
G=beta[1]*np.identity(N)+beta[2]*W
sigma=0.5**abs(np.array([0,1,2,3,4]*5).reshape((5,5))-np.array([0,1,2,3,4]*5).reshape((5,5)).T)
Z=np.random.multivariate_normal(np.ones(5),sigma,N)
def g0(z):
    return(0.2-0.5*z[:,0]+0.3*z[:,1]+0.8*z[:,2]-0.1*z[:,3]-0.1*z[:,4])                       # case 1
    #return(5-2*z[:,0]+0.5*z[:,1]**2-z[:,2]**3-np.log(z[:,3]+3)+np.sqrt(z[:,4]+3))     # case 2
    # return(z[:,0]**2-2*z[:,1]**2+z[:,1]*z[:,2]+z[:,3]*z[:,4])                       # case 3
B0=g0(Z)
mu0=np.linalg.inv((1-beta[1])*np.identity(N)-beta[2]*W).dot(B0)
sig=1
cov0=sig**2*np.linalg.inv(np.identity(N**2)-np.kron(G,G)).dot(np.ravel(np.identity(N))).reshape(N,N).T
Y0=np.random.multivariate_normal(mu0,cov0,1)



from torch.utils.data import DataLoader
simul = 50
cpcout = np.zeros((2,3))
thetazhu = np.zeros((3,simul))
msezhu = np.zeros(simul)

for simu in range(0,simul):
    random.seed(simu)
    Y=np.zeros((N,T*2))
    epsilon=np.random.multivariate_normal(np.zeros(N),sig*np.identity(N),2*T)
    Y[:,0]=Y0
    Y[:,1]=B0+G.dot(Y0.T)[:,0]+epsilon[0,:]
    for i in range(0,2*T-2):
        Y[:,i+2]=B0+G.dot(Y[:,i+1])+epsilon[i+1,:]
    YY=Y[:,int(T-1-T/10):int(2*T-T/10)]    
    YYtest=Y[:,int(2*T-T/10-1):2*T]    
    p=5
    XX=np.zeros((T,N,p+3))
    XXtest=np.zeros((int(T/10),N,p+3))
    CC=np.zeros((p+3,p+3))
    BB=np.zeros((p+3,1))
    info=np.zeros((3,3))
    MSE=0
    for t in range(0,T):
        XX[t,:,0]=np.ones(N)
        XX[t,:,1]=YY[:,t].ravel()
        #XX[t,:,2]=(W*(np.mat(YY[:,t])).T).ravel()
        XX[t,:,2]=W.dot(YY[:,t])
        XX[t,:,3:8]=Z
        CC=CC+np.dot(XX[t,:,:].T,XX[t,:,:])
        BB=BB+XX[t,:,:].T*(np.mat(YY[:,t+1]).T)
    for t in range(0,int(T/10)):
        XXtest[t,:,0]=np.ones(N)
        XXtest[t,:,1]=YYtest[:,t].ravel()
        #XX[t,:,2]=(W*(np.mat(YY[:,t])).T).ravel()
        XXtest[t,:,2]=W.dot(YYtest[:,t])
        XXtest[t,:,3:8]=Z
    thetahat=np.linalg.inv(CC)*BB
    thetazhu[:,simu] = np.ravel(thetahat[0:3,0])
    for t in range(0,T):
        MSE=MSE+np.sum((YY[:,t+1]-np.ones(N)*thetahat[0,0]-np.ravel((Z.dot(thetahat[3:8,0])))-(thetahat[1,0]*np.identity(N)+thetahat[2,0]*W).dot(YY[:,t]))**2)
    for t in range(0,int(T/10)):
        msezhu[simu]=msezhu[simu]+np.sum((YYtest[:,t+1]-np.ones(N)*thetahat[0,0]-np.ravel((Z.dot(thetahat[3:8,0])))-(thetahat[1,0]*np.identity(N)+thetahat[2,0]*W).dot(YYtest[:,t]))**2)
    sig2zhu = MSE/N/T
    se2zhu = np.linalg.inv(CC)*sig2zhu
    if thetahat[0] >= beta[0]-1.96*np.sqrt(se2zhu[0,0]) and thetahat[0] <= beta[0]+1.96*np.sqrt(se2zhu[0,0]):
        cpcout[1,0] += 1
    if thetahat[1] >= beta[1]-1.96*np.sqrt(se2zhu[1,1]) and thetahat[1] <= beta[1]+1.96*np.sqrt(se2zhu[1,1]):
        cpcout[1,1] += 1
    if thetahat[2] >= beta[2]-1.96*np.sqrt(se2zhu[2,2]) and thetahat[2] <= beta[2]+1.96*np.sqrt(se2zhu[2,2]):
        cpcout[1,2] += 1

print(cpcout/simul)
print(np.mean(msezhu))
print(np.std(msezhu), ddof=1)

print(np.mean(thetazhu[0,:]))
print(np.std(thetazhu[0,:]), ddof=1)

print(np.mean(thetazhu[1,:]))
print(np.std(thetazhu[1,:]), ddof=1)

print(np.mean(thetazhu[2,:]))
print(np.std(thetazhu[2,:]), ddof=1)