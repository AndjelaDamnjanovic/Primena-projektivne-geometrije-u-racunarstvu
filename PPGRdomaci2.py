import numpy as np
import sys
from scipy.linalg import svd
import cv2
from matplotlib import pyplot as plt
import imageio
import imutils
import math

def UAfine(dot):
	return [round(dot[0]/dot[2]), round(dot[1]/dot[2]),1]

def nadjiMatricuZaJednoPreslikavanje(A,B,C,D):
	#pravimo matrice od nizova koordinata
	matrica=np.reshape([A,B,C], (3,3)).transpose()
	matrica1=np.reshape([D,B,C], (3,3)).transpose()
	matrica2=np.reshape([A,D,C], (3,3)).transpose()
	matrica3=np.reshape([A,B,D], (3,3)).transpose()
	
	#racunamo determinanate matrica
	delta=np.linalg.det(matrica)
	delta1=round(np.linalg.det(matrica1))
	delta2=round(np.linalg.det(matrica2))
	delta3=round(np.linalg.det(matrica3))
	
	
	#nalazimo vrednosti parametara lambda koristeci Kramerovo pravilo
	lambda1=(delta1/delta)
	lambda2=(delta2/delta)
	lambda3=(delta3/delta)	
	
	#formiramo matricu
	A=np.array(A)*lambda1
	B=np.array(B)*lambda2
	C=np.array(C)*lambda3
	
	
	return np.reshape([A,B,C], (3,3)).transpose()
	
def matricaKorespondencije(A,A1):
	return np.reshape([0,	 	0, 		0, -A1[2]*A[0], -A1[2]*A[1], -A1[2]*A[2], A1[1]*A[0], A1[1]*A[1], A1[1]*A[2],
			A1[2]*A[0], A1[2]*A[1], A1[2]*A[2],    0,		0, 	0, -A1[0]*A[0], -A1[0]*A[1], -A1[0]*A[2],], (2,9))
		
def naivni(A,B,C,D,A1,B1,C1,D1):
	
	#trazimo matrice P1 i P2 preslikavanja u i iz baznih koordinata
	P1=nadjiMatricuZaJednoPreslikavanje(A,B,C,D)
	P2=nadjiMatricuZaJednoPreslikavanje(A1,B1,C1,D1)
	
	#vracamo skalarni proizvod koji odgovara matrici P
	return P2.dot(np.linalg.inv(P1)).round(5)
	

def DLT(A,B,C,D,A1,B1,C1,D1):
	#nalazimo matrice korespondencije dimenzija 2x9
	matrica1=matricaKorespondencije(A,A1)
	matrica2=matricaKorespondencije(B,B1)
	matrica3=matricaKorespondencije(C,C1)
	matrica4=matricaKorespondencije(D,D1)
	
	#nalazimo matricu dimanzaija 8x9 nastalu spajanjem ostalih matrica korespondencije
	matrica=np.reshape([matrica1, matrica2, matrica3, matrica4], (8,9))
	
	#radimo SVD dekompoziciju dobijene matrice
	U, s, VT = svd(matrica)
	
	#vracamo matricu P koja se dobija od poslednje kolone VT[9]
	return np.reshape(VT[8].transpose(), (3,3))


	
def DLTviseTacaka(originali, slike):
	n = len(originali)
	rez = matricaKorespondencije(originali[0], slike[0])
	for i in range(1, len(originali)):
		pom = matricaKorespondencije(originali[i], slike[i])
		rez = np.concatenate((rez, pom))
	U, D, VT = np.linalg.svd(rez)
	V = np.transpose(VT)
	kolona = np.reshape(V[-1], (3, 3)).round(5)
	return kolona
	
def normalizacija(originali):
	#racunamo teziste svih tacaka
	Tx = sum([el[0] / el[2] for el in originali]) / len(originali)
	Ty = sum([el[1] / el[2] for el in originali]) / len(originali)
	teziste=[Tx, Ty]
	dist = 0.0
	#racunamo prosecno rastojanje
	for i in range(0, len(originali)):
		a = float(originali[i][0] / originali[i][2]) - teziste[0]
		b = float(originali[i][1] / originali[i][2]) - teziste[1]

		dist += math.sqrt(a ** 2 + b ** 2)

	dist = dist / float(len(originali))

	ro = float(math.sqrt(2)) / dist
	#racunamo matricu preslikavanja
	T = np.array([[ro, 0, -ro * teziste[0]], [0, ro, -ro * teziste[1]], [0, 0, 1]])
	return T


def DLTModifikovan(originali, slike):
	#trazimo matrice koje normalizuju tacke originala i slika
	T = normalizacija(originali)
	Tp = normalizacija(slike)
	
	#nalazimo normalizovane koordinate tacaka
	N = T.dot(np.transpose(originali))
	Np = Tp.dot(np.transpose(slike))

	N = np.transpose(N)
	Np = np.transpose(Np)
	
	#nalazimo matricu dlt algoritma za norm tacke
	P1 = DLTviseTacaka(N, Np)

	P = (np.linalg.inv(Tp)).dot(P1).dot(T)
	return P
	
	
#RANSAC algoritam
def homography(kpA, kpB, featureA, featureB, matches, reprojectionThreshold):
	kpA=np.float32([kp.pt for kp in kpA])
	kpB=np.float32([kp.pt for kp in kpB])
	
	if len(matches)>4:
		ptsA=np.float32([kpA[m.queryIdx] for m in matches])
		ptsB=np.float32([kpB[m.queryIdx] for m in matches])
		
		(H, status)=cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojectionThreshold)
		
		return (matches, H, status)
	else:
		return None
	

A=[2,0,1]
B=[-2,1,1]
C=[-1, -4, 1]
D=[0,2, 1]
E=[2,2, 1]
F=[-8,-2,1]
G=[1,2,3]

A1=[-2,1,1]
B1=[2, -1,1]
C1=[1, -2,1]
D1=[3, -1,1]
E1=[-12,1,1]
F1=[-16, -5, 4]
G1=[12, 1, 1]

odg=int(input("Unesite redni broj algoritma koji zelite da primenite: 1 za naivni, 2 za DLT, 3 za modifikovani DLT, 4 za ransac: "))

A=UAfine(A)
B=UAfine(B)
C=UAfine(C)
D=UAfine(D)
E=UAfine(E)
F=UAfine(F)


A1=UAfine(A1)
B1=UAfine(B1)
C1=UAfine(C1)
D1=UAfine(D1)
E1=UAfine(E1)
F1=UAfine(F1)


originali=[A,B,C,D,E]
slike=[A1,B1,C1,D1,E1]

	
if odg<=0 or odg>4:
	print("Naveli ste neregularan identifikator!!!")
	sys.exit()
elif odg!=4:
	
	odgovor=input("Da li zelite da unesete svoje koordinate tacaka? [da/ne]")
	if odgovor=="da" and (odg==1 or odg==2):
		A=(input("Unesite homogene koordinate tacke: ")).split(' ')
		A=UAfine([int(A[0]), int(A[1]), int(A[2])])
		B=(input("Unesite homogene koordinate tacke: ")).split(' ')
		B=UAfine([int(B[0]), int(B[1]), int(B[2])])
		C=(input("Unesite homogene koordinate tacke: ")).split(' ')
		C=UAfine([int(C[0]), int(C[1]), int(C[2])])
		D=(input("Unesite homogene koordinate tacke: ")).split(' ')
		D=UAfine([int(D[0]), int(D[1]), int(D[2])])
		print("Sada unesite koordinate slika: ")
		A1=(input("Unesite homogene koordinate tacke: ")).split(' ')
		A1=UAfine([int(A1[0]), int(A1[1]), int(A1[2])])
		B1=(input("Unesite homogene koordinate tacke: ")).split(' ')
		B1=UAfine([int(B1[0]), int(B1[1]), int(B1[2])])
		C1=(input("Unesite homogene koordinate tacke: ")).split(' ')
		C1=UAfine([int(C1[0]), int(C1[1]), int(C1[2])])
		D1=(input("Unesite homogene koordinate tacke: ")).split(' ')
		D1=UAfine([int(D1[0]), int(D1[1]), int(D1[2])])
	elif odgovor not in {"da", "ne"}:
		print("Greska!!! Morate uneti da ili ne")
		sys.exit()
	elif odgovor=="da" and odg==3:
		n=int(input("KOliko tacaka zelite da unesete? "))
		originali=[]
		slike=[]
		for i in range(n):
			tacka=(input("Unesite homogene koordinate tacke: ")).split(' ')
			tacka=UAfine([int(tacka[0]), int(tacka[1]), int(tacka[2])])
			originali.append(tacka)
		print("Sada unesite koordinate slika: ")
		for i in range(n):
			tacka=(input("Unesite homogene koordinate tacke: ")).split(' ')
			tacka=UAfine([int(tacka[0]), int(tacka[1]), int(tacka[2])])
			slike.append(tacka)
else:
	print("Odlucili ste se za panoramu.")

if odg==1:
	P=naivni(A,B,C,D,A1,B1,C1,D1)
	print(P.round(4))
	P=np.multiply(P, 1/P[0,0])
	print(P.round(4))
elif odg==2:
	P=DLT(A,B,C,D,A1,B1,C1,D1)
	#P=DLTviseTacaka(originali, slike)
	print(P)
	P=np.multiply(P, 1/P[0,0])
	print("Matrica DLT algoritma:")
	print(P.round(4))
	
	A=[A[0], A[1]+2, A[2]]
	P=DLT(A,B,C,D,A1,B1,C1,D1)
	print(P)
	P=np.multiply(P, 1/P[0,0])
	print("Matrica DLT algoritma:")
	print(P.round(4))
	
elif odg==3:
	P=DLTModifikovan(originali, slike)
	P=np.multiply(P, 1/P[0,0])
	print("Matrica modifikovanog DLT algoritma sa unetim brojem tacaka:")
	print(P.round(4))
	#print("Ukoliko koristimo modifikovani DLT algoritam, matrica preslikavanja je invarijantna na promenu koordinata, sto nije slucaj sa obicnim DLT algoritmom!")
	
	#A=[A[0], A[1]+2, A[2]]
	#originali=[A,B,C,D,E]
	#slike=[A1,B1,C1,D1,E1]
	#P=DLTModifikovan(originali, slike)
	#P=np.multiply(P, 1/P[0,0])
	#print("Matrica modifikovanog DLT algoritma sa unetim brojem tacaka:")
	#print(P.round(4))
	
else:
	#spajanje slika automatskim detektovanjem tacaka koje se poklapaju
	images=[]
	image = cv2.imread("slika1.png")
	images.append(image)
	image = cv2.imread("slika2.png")
	images.append(image)
	
	
	stitcher = cv2.Stitcher_create()
	(status, stitched) = stitcher.stitch(images)
		
		
	if status == 0:
		# display the output stitched image to our screen
		cv2.imshow("Stitched", stitched)
		cv2.waitKey(0)
		# otherwise the stitching failed, likely due to not enough keypoints)
		# being detected
	else:
		print("[INFO] image stitching failed ({})".format(status))
	
	#spajanje slika tako sto korisnik unese misem tacke koje se poklapaju
	
	image2=cv2.imread("slika2.png")
	image2_greyscale=cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
	
	image1=cv2.imread("slika1.png")
	image1_greyscale=cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
	
	fig, (ax, ay)=plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(15, 10))
	ax.imshow(image1, cmap="gray")
	ay.imshow(image2, cmap="gray")
	
	plt.show()
	
	#detekcija tacaka
	descriptor = cv2.SIFT_create()
	#kp = sift.detect(image2_greyscale,None)
	#image2=cv2.drawKeypoints(image2_greyscale,kp,image2)
	#cv2.imshow('slika', image2)
	#cv2.waitKey(0)
	kpA, featureA=descriptor.detectAndCompute(image2_greyscale, mask=None)
	kpB, featureB=descriptor.detectAndCompute(image1_greyscale, mask=None)
	
	def bf_match(featureA, featureB):
		bf=cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
		
		best_matches=bf.match(featureA, featureB)
		
		rawMatches=sorted(best_matches, key=lambda x: x.distance)
		
		return rawMatches
	fig=plt.figure(figsize=(20, 10))
	
	#povezane tacke
	matches=bf_match(featureA, featureB)
	
	final_image=cv2.drawMatches(image2, kpA, image1, kpB, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	plt.imshow(final_image)
	plt.show()
	
	(_, H, _)=homography(kpA, kpB, featureA, featureB, matches, reprojectionThreshold=4)
	print(H)
	
	#transformisanje
	
	width=image1.shape[1]+image2.shape[1]
	height=image1.shape[0]+image2.shape[0]
	
	res=cv2.warpPerspective(image2, H, (height, width))
	res[0:image1.shape[0], 0:image1.shape[1]]=image1

	grey=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
	threshold=cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY)[1]
	
	cnts=cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts=imutils.grab_contours(cnts)
	
	c=max(cnts, key=cv2.contourArea)
	(x,y,w,h)=cv2.boundingRect(c)
	
	res=res[y:y+h, x:x+w]
	plt.figure(figsize=(20,10))
	plt.imshow(res)
	plt.show()
#print("DLP algoritam sa 5 tacaka: ")
#P=DLTviseTacaka(originali, slike)
#P=np.multiply(P, 1/P[0,0])
#print(P.round(4))
