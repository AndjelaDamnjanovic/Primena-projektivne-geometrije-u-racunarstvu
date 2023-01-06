import sys
import math
import numpy as np
from matplotlib.animation import FuncAnimation

from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *

def Xrotation(fi):
	sinus=math.sin(fi)
	cosinus=math.cos(fi)
	return np.reshape([1,0,0,0,cosinus, -sinus, 0,sinus, cosinus], (3,3))
	
def Yrotation(teta):
	sinus=math.sin(teta)
	cosinus=math.cos(teta)
	return np.reshape([cosinus, 0, sinus, 0,1,0,-sinus,0,cosinus], (3,3))
	
def Zrotation(psi):
	sinus=math.sin(psi)
	cosinus=math.cos(psi)
	return np.reshape([cosinus,-sinus, 0, sinus, cosinus, 0, 0,0,1], (3,3))
	
def normalizuj(p):
	norma=(np.linalg.norm(p))
	if norma:
		return [p[0]/norma, p[1]/norma, p[2]/norma]
	else:
		print("Deljenje nulom!!!")
	
def isIdentity(A):
	for i in range(3):
		for j in range(3):
			if (A[i][j]!=1 and i==j) or (A[i][j]!=0 and i!=j):
				return False
	return True
	
def normalizujQ(q):
	norma=np.linalg.norm(q)
	return [q[0]/norma, q[1]/norma, q[2]/norma, q[3]/norma]

def Euler2A(psi, theta, phi):
	Rz = Zrotation(psi)
	Ry = Yrotation(theta)
	Rx = Xrotation(phi)
	
	return np.matmul(Rz, np.matmul(Ry, Rx))	
	#return np.matmul(np.matmul(Rz, Ry), Rx)

def AxisAngle(A):
	A = np.array(A)
	
	#proveravamo da li je matrica ortogonalna i napustamo program ako nije
	if (round(np.linalg.det(A)) != 1):
		sys.exit("Matrix is not orthogonal")
	#nalazimo matricu A-E
	Ap = A - np.identity(3)
	koef=A[0][0]/Ap[0][0]
	
	
	#ispitujemo da li su prva dva vektora linearno zavisni
	if(Ap[0][0] == Ap[1][0]*koef and Ap[0][1] == Ap[1][1]*koef and Ap[0][2] == Ap[1][2]*koef):
		#ako jesu, racunamo vektorski proizvod prve sa poslednjom vrstom
		g = np.cross(Ap[0], Ap[2])
	else:
		#inace, racunamo njihov vektorski proizvod
		g = np.cross(Ap[0], Ap[1])
		
	#sada trazimo vektor h koji je normalan na g
	h = [0, 0, 0]
	h[0] = g[1]
	h[1] = -g[0]
	
	h = np.array(h)
	
	#normalizujemo h
	h = normalizuj(h)
	
	#sada trazimo vektor hp=A*h 
	hp = np.matmul(A, h)
	
	#racunamo ugao koji vektori h i hp zaklapaju koriscenjem skalarnog proizvoda
	ugao = math.acos(np.dot(h, hp))
	
	#racunamo mesoviti proizvod vektora [h,hp,g]
	mesoviti = [[h[0], h[1], h[2]],[hp[0], hp[1], hp[2]],[g[0], g[1], g[2]]]
	mesoviti = np.array(mesoviti)
	
	#ako je determinanta mesovitog proizvoda manja od 0, onda smo uzeli vektor sa pogresnim usmerenjem
	if (np.linalg.det(mesoviti) < 0):
		g = -1 * g
	g = normalizuj(g)

	return g, ugao	
	
def Rodrigez(p, fi):
	p=normalizuj(p)
	ppt = np.outer(p, np.transpose(p))
	px=np.reshape([0, -p[2], p[1], p[2], 0, -p[0], -p[1], p[0], 0], (3,3))

	cosinus=np.cos(fi).round(5)
	sinus=np.sin(fi).round(5)
	return np.multiply(ppt,(1-cosinus))+np.multiply(np.identity(3), cosinus)+np.multiply(px, sinus)
	
def A2Euler(A):
	if round(np.linalg.det(A)) !=1:
		sys.exit("Greska!!! Determinanta matrice nije 1!")
	if A[2][0] not in {1,-1}:
		psi=math.atan2(A[1][0], A[0][0])
		theta=np.arcsin(-A[2][0])
		phi=math.atan2(A[2][1], A[2][2])
	elif A[2][0]==1:
		theta=-math.pi/2
		phi=0
		psi=math.pi
	else:
		theta=math.pi/2
		phi=0
		psi=math.pi
	return [phi, theta, psi]
	
    	
def AxisAngle2Q(p,fi):
	p=normalizuj(p)
	
	ugao=math.sin(fi/2)
	
	qx=p[0]*ugao
	qy=p[1]*ugao
	qz=p[2]*ugao
	qw=math.cos(fi/2)
	
	return [qx, qy, qz, qw]
	
def Q2AxisAngle(q):
	q = normalizujQ(q)
	if(q[3] < 0):
		q = -1 * q

	ugao = 2 * np.arccos(q[3])

	if(np.abs(q[3]) == 1):
		qp = np.array([1, 0, 0])
	else:
        	qp = normalizuj([q[0], q[1], q[2]])
	return qp,ugao
	
def slerp(q1, q2, tm, t):
	if t < 0 or t > tm:
		sys.exit("t nije u odgovarajucem opsegu")

	norm_q1 = np.linalg.norm(q1)
	if norm_q1 != 0:
		q1 = q1 / norm_q1

	norm_q2 = np.linalg.norm(q2)
	if norm_q2 != 0:
		q2 = q2 / norm_q2

	cos0 = np.dot(np.array(q1), np.array(q2))

	if cos0 < 0:
		q1 = -q1
		cos0 = -cos0

	if cos0 > 0.95:
		return q1

	fi0 = math.acos(cos0)

	return ((math.sin(fi0 * (1 - t/tm))) / math.sin(fi0))*q1 + ((math.sin(fi0 * (t/tm))) / math.sin(fi0))*q2
	
psi=math.pi/4*(59%7+1)
teta=math.pi/17*(59%8+1)
fi=math.pi/3*(59%5+1)


print("Psi: ", psi, "\nTeta: ", teta, "\nFi: ", fi)

print("\nEuler2A algoritam:\n", )
A=Euler2A(fi,teta,psi)
A=A.round(4)
print(A, '\n')
print('--------------------------------------------------------------------------------------------------')

print("Axis angle:\n")
vektor, ugao=AxisAngle(A)
print("Vektor je: ", vektor, ", a ugao je: ", ugao, '\n')
print('--------------------------------------------------------------------------------------------------')

print("Rodrigez:\n")
R=Rodrigez(vektor, ugao)
R=R.round(4)
print(R, '\n')
print('--------------------------------------------------------------------------------------------------')

print("A2Euler:\n")
vec=A2Euler(A)
print("Uglovi su: ", vec,  '\n')
print("Ugao fi koji izbacuje program jednak je uglu fi koji je unet, samo je umesto 300 stepeni napisan kao -60")
print('--------------------------------------------------------------------------------------------------')

print("AxisAngle2Q:\n")
q=AxisAngle2Q(vektor, ugao)
print(q, '\n')
print('--------------------------------------------------------------------------------------------------')

print("Q2AxisAngle:\n")
vektor, ugao=Q2AxisAngle(q)
print("Vektor: ", vektor, ", ugao: ", ugao, '\n')
print('--------------------------------------------------------------------------------------------------')

#class Globals:
#	def __init__(self):
#		self.timer_active = False
		
		#inicijalizujemo informacije koje su nam potrebne za SLerp fju
		#self.t = 0
		#self.tm = 40
		#self.q1 = []
		#self.q2 = []
		
		#pocetna orijentacija
		#self.fi1 = math.pi/6
		#self.teta1 = math.pi/2
		#self.psi1 = math.pi/4
        	
        	#krajnja orijentacija
		#self.fi2 = fi
		#self.teta2 = teta
		#self.psi2 = psi
		
		#pocetne koordiante
		#self.x1 = 0
		#self.y1 = 0
		#self.z1 = 0
		
		#krajnje koordiante
		#self.x2 = -3
		#self.y2 = 2
		#self.z2 = 4


#g = Globals()


#def onDisplay():
#	global g

	# parametri svetla
#	light_position = [0, 7, 0, 0]
#	light_ambient = [0.5, 0.5, 0.5, 0.1]
#	light_diffuse = [0.8, 0.8, 0.8, 1]
#	light_specular = [0.6, 0.6, 0.6, 1]
#	shininess = 30
	
	#praznimo bafere za boju i dubinu
#	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

	# podesava se tacka pogleda
#	glMatrixMode(GL_MODELVIEW)
#	glLoadIdentity()
#	gluLookAt(8, 10, 8,
 #             0, 0, 0,
  #            0, 1, 0)

	# objekti zadrzavaju boju
#	glEnable(GL_COLOR_MATERIAL)
#	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

	# podesava se svetlo
#	glEnable(GL_LIGHTING)
#	glEnable(GL_LIGHT0)

#	glLightfv(GL_LIGHT0, GL_POSITION, light_position)
#	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
#	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
#	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
#	glMaterialf(GL_FRONT, GL_SHININESS, shininess)

	# Iscrtava se koordinatni sistem
#	drawCoordinateSystem(20)

	# Iscrtavaju se pocetni i krajnji objekat
#	drawBeginAndEnd(g.x1, g.y1, g.z1, g.fi1, g.teta1, g.psi1, 0, 0.75, 1)
#	drawBeginAndEnd(g.x2, g.y2, g.z2, g.fi2, g.teta2, g.psi2, 0, 0.75, 1)

	# Iscrtava se pomereni objekat u trenutku t
#	drawAnimated(0, 0.75, 1)

#	glutSwapBuffers()


#def drawCoordinateSystem(size):
#	glBegin(GL_LINES)
#	glColor3f(1, 0, 0)
#	glVertex3f(0, 0, 0)
"""
	glVertex3f(size, 0, 0)

	glColor3f(0, 1, 0)
	glVertex3f(0, 0, 0)
	glVertex3f(0, size, 0)

	glColor3f(0, 0, 1)
	glVertex3f(0, 0, 0)
	glVertex3f(0, 0, size)
	glEnd()


def drawBeginAndEnd(x, y, z, fi, teta, psi, r, g, b):
	glPushMatrix()

	glColor3f(r, g, b)
	glTranslatef(x, y, z)

	A = Euler2A(fi, teta, psi)
	p, alpha = AxisAngle(A)

	glRotatef(alpha / math.pi * 180, p[0], p[1], p[2])

	drawCoordinateSystem(4)

	glPopMatrix()


def drawAnimated(r, gr, b):
	global g

	glPushMatrix()
	glColor3f(r, gr, b)
	
	#SLERP DEO 

	x = (1 - g.t/g.tm)*g.x1 + (g.t/g.tm)*g.x2
	y = (1 - g.t / g.tm) * g.y1 + (g.t / g.tm) * g.y2
	z = (1 - g.t / g.tm) * g.z1 + (g.t / g.tm) * g.z2

	glTranslatef(x, y, z)

	q = slerp(g.q1, g.q2, g.tm, g.t)
	p, fi = Q2AxisAngle(q)

	glRotatef(fi / math.pi * 180, p[0], p[1], p[2])


	drawCoordinateSystem(4)

	glPopMatrix()


def onReshape(w, h):
	glViewport(0, 0, w, h)

	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	gluPerspective(60, float(w) / h, 1, 1500)


def onKeyboard(ch, x, y):
	global g

	if ord(ch) == 27:
		sys.exit(0)
	elif ord(ch) == ord('g') or ord(ch) == ord('G'):
		if not g.timer_active:
			glutTimerFunc(100, onTimer, 0)
			g.timer_active = True
	elif ord(ch) == ord('s') or ord(ch) == ord('S'):
		g.timer_active = False


def onTimer(value):
	global g

	if value != 0:
		return

	g.t += 1
	if g.t > g.tm:
		g.t = 0
		g.timer_active = False
		return

	glutPostRedisplay()

	if g.timer_active:
		glutTimerFunc(100, onTimer, 0)


def main():
	global g

	glutInit()
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE)
	glutInitWindowSize(800, 600)
	glutInitWindowPosition(0, 0)
	glutCreateWindow("SLERP animacija")

	glutDisplayFunc(onDisplay)
	glutReshapeFunc(onReshape)
	glutKeyboardFunc(onKeyboard)

	glEnable(GL_DEPTH_TEST)

	# inicijalizuje se pocetni kvaternion q1
	A = Euler2A(g.fi1, g.teta1, g.psi1)
	p, alpha = AxisAngle(A)
	g.q1 = AxisAngle2Q(p, alpha)

	# inicijalizuje se krajnji kvaternion q2
	A = Euler2A(g.fi2, g.teta2, g.psi2)
	p, alpha = AxisAngle(A)
	g.q2 = AxisAngle2Q(p, alpha)

	glutMainLoop()
	


if __name__ == '__main__':
	main()
"""

