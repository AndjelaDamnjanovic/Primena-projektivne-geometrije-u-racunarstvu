import sys
import numpy as np
from scipy.linalg import qr

from matplotlib import pyplot as plt 
from matplotlib import image as mpimg

def lepIspis(dot):
	return [round(dot[0]), round(dot[1]),1]
	
def prebaciIzAfinih(dots):
	newList=[]
	for dot in dots:
		newDot=[dot[0], dot[1], 1]
		newList.append(newDot)
	return newList


def ParametriKamere(T):
	
	#nalzaimo T0
	T0=np.transpose(T)
	T0=np.transpose(T0[:3])
	
	if np.linalg.det(T0)<0:
		T*=-1
	
	Q,R=qr(np.linalg.inv(T0))
	
	#ako ima negativnih vrednosti an dijagonali u matrici R
	for i in range(R.shape[1]):
		if R[i][i]<0:
			R[i][:]*=-1
			Q[0][i]*=-1
			Q[1][i]*=-1
			Q[2][i]*=-1
	
	#K je inverz od R
	K=np.linalg.inv(R)
	
	#delimo elementom K33
	K*=1/K[2][2]
	
	#A je transponovana matrica Q
	A=np.transpose(Q)
	
	#trazimo C
	C=[]
	for i in range(4):
		#nalazimo minore matrice T
		#np.delete radi tako sto iz matrice T redom brise jednu po jednu kolonu jer je indikator 1
		tmp_T=np.delete(T,i,1)
		C.append((-1)**i* np.linalg.det(tmp_T))
		
	#prebacujemo C u np.array
	C=np.array(C)
	C/=C[3]
	C=C[:-1]
	return K,A,C
	
def matricaKorespondencije(A,A1):
	return np.reshape([0, 0, 0, 0,
	-A1[2]*A[0], -A1[2]*A[1], -A1[2]*A[2], -A1[2]*A[3], 
	A1[1]*A[0], A1[1]*A[1], A1[1]*A[2],A1[1]*A[3], 
	A1[2]*A[0], A1[2]*A[1], A1[2]*A[2] ,A1[2]*A[3],
	 0, 0, 0, 0, 
	 -A1[0]*A[0], -A1[0]*A[1], -A1[0]*A[2], -A1[0]*A[3]], 
	 (2,12))
		

def DLTviseTacaka(originali, slike):
	n = len(originali)
	rez = matricaKorespondencije(originali[0], slike[0])
	for i in range(1, len(originali)):
		pom = matricaKorespondencije(originali[i], slike[i])
		rez = np.concatenate((rez, pom))
	np.reshape(rez,(2*n, 12))
	U, D, VT = np.linalg.svd(rez)

	#treba nam samo poslednja kolona matrice VT
	T = VT[-1]
	T = T.reshape(3, 4)

	T /= T[0][0]
	return T
	

def CameraDLP(originali, slike):
	if len(originali)!=len(slike):
		sys.exit("Nizovi originala i slika nisu iste duzine!!!")
	if len(originali)<6:
		sys.exit("Nema dovoljan broj korespodencija. Minimalan dovoljan broj korespondencija je 6!")
	
	rez=DLTviseTacaka(originali, slike)
	print("Matrica T:\n")
	print(rez)
	return [rez[0], rez[1], rez[2]]
n=9

T=[[5, -1-2*n, 3, 18-3*n], 
   [0,   -1,   5,  21],
   [0,   -1,   0,   1]]
   
   
   
M1 = [460, 280, 250, 1]
M2 = [50, 380, 350, 1]
M3 = [470, 500, 100, 1]
M4 = [380, 630, 50*n, 1]
M5 = [30*n, 290, 0, 1]
M6 = [580, 0, 130, 1]

M1p = [288, 251, 1]
M2p = [79, 510, 1]
M3p = [470, 440, 1]
M4p = [520, 590, 1]
M5p = [365, 388, 1]
M6p = [365, 20, 1]

originali = [M1, M2, M3, M4, M5, M6]
slike = [M1p, M2p, M3p, M4p, M5p, M6p]

   
K,A,C=ParametriKamere(T)

print("n=", n)
print("------------------------------------------------------------------------")
print("a) Parametri kamere su:\n")
print("Polazna matrica T:\n", np.reshape(T, (3,4)))
print("Matrica K:\n", K)
print("Matrica A:\n", A)
print("Centar C:\n ", C)

print("------------------------------------------------------------------------")
print("b) Matrica kamere je:\n")
T=CameraDLP(originali, slike)

print("------------------------------------------------------------------------")
print("c)\n")


image=mpimg.imread("domaci4(1)(1).jpg")
fig, ax=plt.subplots()
ax.imshow(image)

print("Unesite 6 tacaka misem: ")

while(True):
	yroi=plt.ginput(n=6)
	break
newList=prebaciIzAfinih(yroi)

i=1
print("Uneli ste sledece koordinate:")

slike=[M1p, M2p, M3p, M4p, M5p, M6p]
i=0
for item in newList:
	slike[i]=item
	print("T",i+1,  ": ", lepIspis(item))
	i+=1
	
	
#izmerene koordinate u mm:
M1 = [30, 655, 197, 1]
M2 = [423, 572, 14, 1]
M3 = [102, 314, 50, 1]
M4 = [102, 243, 0, 1]
M5 = [27, 102, 0, 1]
M6 = [99, 28, 33, 1]
originali=[M1, M2, M3, M4, M5, M6]
#CameraDLP(originali, slike)
K,A,C=ParametriKamere(CameraDLP(originali, slike))
print("Matrica K:\n", K)
print("Matrica A:\n", A)
print("Centar C:\n ", C)
