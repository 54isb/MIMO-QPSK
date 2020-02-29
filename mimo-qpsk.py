import numpy as np
import warnings
import itertools

warnings.filterwarnings('ignore')
DET = 0 #CHOOSE DETECTOR: 0: ZF, 1: MMSE, 2: ML
N = 4 #number of Tx antennas (=data streams)
M = 4 #number of Rx antenaas
ebn0_max = 12 #maximum EbN0
EbN0 = 0.0 #initial EbN0
EbN0_Width = 2.0 #EbN0 interval
itermax = 100000 #num. of simulations

def mapping(x): #NOTE Gray-labeled QPSK
    condlist = [x ==0, x==1, x==2, x==3]
    funclist = [lambda x: 1.0/np.sqrt(2.0) + 1j * 1.0/np.sqrt(2.0),lambda x: -1.0/np.sqrt(2.0) + 1j*1.0/np.sqrt(2.0),lambda x:1.0/np.sqrt(2.0) - 1j * 1.0/np.sqrt(2.0),lambda x:-1.0/np.sqrt(2.0) - 1j * 1.0/np.sqrt(2.0)]
    return np.piecewise(x.astype(dtype=np.complex), condlist, funclist)

def demapping(x):
    condlist = [(np.real(x)>0.0)&(np.imag(x)>0.0),(np.real(x)<=0.0)&(np.imag(x)>0.0),(np.real(x)>0.0)&(np.imag(x)<=0.0),(np.real(x)<=0.0)&(np.imag(x)<=0.0)]
    funclist = [lambda x: 0,lambda x: 1,lambda x: 2,lambda x: 3]
    return np.piecewise(x.astype(dtype=np.complex), condlist, funclist)

def comparebit(a,b):
    return np.bitwise_and(1, np.bitwise_xor(a,b)) + np.bitwise_and(1, np.right_shift(np.bitwise_xor(a,b),1))

#Seed generation for RVs
np.random.seed()
#Candidate Generation for ML
x_candidate = mapping(np.array(list(itertools.product(np.array([0,1,2,3]),repeat=N))))

for ebn0 in range(ebn0_max):
    error = 0
    #noise variance for 2-dim Gaussian r.v.
    sigma = np.sqrt(1.0/np.power(10.0,EbN0/10.0) * 0.5 * N)
    #Fundamental SNR (denoting rho)
    SNR = 2.0 * np.power(10.0,EbN0/10.0)

    #repeating trials until itermax
    for k in range(itermax):
        #Generate Discrete Signals
        data = np.random.randint(0,4,(N,1))
        #Mapping Data onto QPSK symbols
        x0 = mapping(data)
        #Channel matrix (M x N), iid Complex Gaussian w/ mean 0, var 1
        H = np.random.normal(0, np.sqrt(0.5), (M,N)) + np.random.normal(0, np.sqrt(0.5), (M,N)) * 1j

        #n observation signals
        y = np.dot(H,x0) + np.random.normal(0.0,np.sqrt(0.5)*sigma,(M,1)) +  np.random.normal(0.0,np.sqrt(0.5)*sigma,(M,1)) * 1j

        if (DET == 0):
            #ZF *************
            s = np.linalg.solve(np.dot(np.transpose(np.conjugate(H)),H),np.dot(np.transpose(np.conjugate(H)),y))
            ans = demapping(s).astype(dtype=np.int)
        elif (DET == 1):
            #Linear MMSE *************
            s = np.linalg.solve(np.dot(np.transpose(np.conjugate(H)),H) + N/SNR * np.identity(N),np.dot(np.transpose(np.conjugate(H)),y))
            ans = demapping(s).astype(dtype=np.int)
        else:
            #ML *************
            ans = demapping(x_candidate[np.argmin(np.linalg.norm(y - np.dot(H,x_candidate.transpose()),axis=0))]).astype(dtype=np.int)

        #Error Count *************
        error += comparebit(ans.reshape(N,1),data).sum()

    #stop if transmission becomes error-free
    if error == 0:
        break

    print('%f,%e' % (EbN0,error/(2.0*itermax*N)))
    EbN0 += EbN0_Width
