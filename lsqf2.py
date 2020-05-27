import numpy as np

def lsqf2(phiIn, magIn, X, Y):
	M1 = np.zeros((3,3))
	M2 = np.zeros((3,3))
	M3 = np.zeros((3,3))
	Md = np.zeros((3,3))
	
	magSq = np.multiply(magIn,magIn)
	
	M1[0,0] = np.sum(np.multiply(magSq, phiIn))
	M1[1,0] = np.sum(np.multiply(np.multiply(magSq, phiIn),X))
	M1[2,0] = np.sum(np.multiply(np.multiply(magSq, phiIn),Y))
	M1[0,1] = np.sum(np.multiply(magSq, X))
	M1[1,1] = np.sum(np.multiply(magSq, np.multiply(X,X)))
	M1[2,1] = np.sum(np.multiply(magSq,np.multiply(X,Y)))
	M1[0,2] = np.sum(np.multiply(magSq, Y))
	M1[1,2] = np.sum(np.multiply(magSq,np.multiply(Y,X)))
	M1[2,2] = np.sum(np.multiply(magSq, np.multiply(Y,Y)))
	
	M2[0,0] = np.sum(magSq)
	M2[1,0] = np.sum(np.multiply(magSq, X))
	M2[2,0] = np.sum(np.multiply(magSq, Y))
	M2[0,1] = np.sum(np.multiply(magSq, phiIn))
	M2[1,1] = np.sum(np.multiply(magSq, np.multiply(phiIn,X)))
	M2[2,1] = np.sum(np.multiply(magSq,np.multiply(phiIn,Y)))
	M2[0,2] = np.sum(np.multiply(magSq, Y))
	M2[1,2] = np.sum(np.multiply(magSq,np.multiply(Y,X)))
	M2[2,2] = np.sum(np.multiply(magSq, np.multiply(Y,Y)))
	
	M3[0,0] = np.sum(magSq)
	M3[1,0] = np.sum(np.multiply(magSq, X))
	M3[2,0] = np.sum(np.multiply(magSq, Y))
	M3[0,1] = np.sum(np.multiply(magSq, X))
	M3[1,1] = np.sum(np.multiply(magSq, np.multiply(X,X)))
	M3[2,1] = np.sum(np.multiply(magSq,np.multiply(X,Y)))
	M3[0,2] = np.sum(np.multiply(magSq, phiIn))
	M3[1,2] = np.sum(np.multiply(magSq,np.multiply(phiIn,X)))
	M3[2,2] = np.sum(np.multiply(magSq, np.multiply(phiIn,Y)))
	
	Md[0,0] = np.sum(magSq)
	Md[1,0] = np.sum(np.multiply(magSq, X))
	Md[2,0] = np.sum(np.multiply(magSq, Y))
	Md[0,1] = np.sum(np.multiply(magSq, X))
	Md[1,1] = np.sum(np.multiply(magSq, np.multiply(X,X)))
	Md[2,1] = np.sum(np.multiply(magSq,np.multiply(X,Y)))
	Md[0,2] = np.sum(np.multiply(magSq, Y))
	Md[1,2] = np.sum(np.multiply(magSq,np.multiply(Y,X)))
	Md[2,2] = np.sum(np.multiply(magSq, np.multiply(Y,Y)))
	
	phi0 = np.linalg.det(M1)/np.linalg.det(Md)
	alpha = np.linalg.det(M2)/np.linalg.det(Md)
	beta = np.linalg.det(M3)/np.linalg.det(Md)
	
	return phi0, alpha, beta

