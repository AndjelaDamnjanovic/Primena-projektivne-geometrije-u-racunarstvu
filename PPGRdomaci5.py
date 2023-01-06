import numpy as np

# Ukljucivanje modula za animaciju
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# Ukljucivanje modula za 3D rekonstrukciju
import numpy as np
import numpy.linalg as LA

#fja preuzeta iz prvog domaceg
def UAfine(dot):
	return (dot/dot[-1])[:-1]

# Normalizacija tacaka
def normalizuj(tacke):
  # Afinizacija tacaka- fja uzeta iz prvog domaceg
  for i in range(len(tacke)):
    tacke[i]= [(tacke[i][0]/tacke[i][2]), (tacke[i][1]/tacke[i][2]),1]
  
  # Teziste tacaka
  tez = np.mean(tacke, axis = 0)

  # Matrica translacije
  g = np.array([[  1,   0, -tez[0]],
                [  0,   1, -tez[1]],
                [  0,   0,    1  ]])


  # Transliranje svih tacaka=mnozenje svih tacaka matricom translacije
  for i in range(len(tacke)):
    tacke[i]=(g@tacke[i])

  # Prosek rastojanja-I OVO MOZE TVOJA FJA

  rast=[]

  for i in range(len(tacke)):
    rast.append(np.sqrt(tacke[i][0]**2+tacke[i][1]**2))
  rast=np.mean(rast)
  
  # Matrica homotetije
  h = np.array([[np.sqrt(2)/rast,         0,           0     ],
                [       0,         np.sqrt(2)/rast,    0     ],
                [       0,                0,           1     ]])

  # Skaliranje svih tacaka;
  # i vracanje skaliranih tacaka i transformacije
  for i in range(len(tacke)):
    tacke[i]=h@tacke[i]


  return h@g, tacke

def normalizujJednu(tacka):
    return tacka/tacka[-1]

#fja koja racuna vektorski proizvod, uzeta sa prvog domaceg
def crossProduct(first, second):
	return (first[1]*second[2]-first[2]*second[1], (-1)*(first[0]*second[2]-first[2]*second[0]), first[0]*second[1]-first[1]*second[0])

def px(p):
  return np.array([[  0,   -p[2],  p[1]],
                            [ p[2],   0,   -p[0]],
                            [-p[1],  p[0],   0  ]])
# Fja za 3D rekonstrukciju
def rekonstruisi():
  #rekonstrukcija iz ravanskih projekcija
  
  # Piksel koordinate vidljivih tacaka (osam), na
  # osnovu kojih se odredjuje fundamentalna matrica
  #x su koordinate leve slike, y desne

  x5  = np.array([562, 494, 1.])
  y5  = np.array([387, 583, 1.])
  x6  = np.array([640, 294, 1.])
  y6  = np.array([626, 487, 1.])
  x9  = np.array([585, 353, 1.])
  y9  = np.array([405, 384, 1.])
  x10  = np.array([671, 128, 1.])
  y10  = np.array([666, 286, 1.])
  x17 = np.array([448, 195, 1.])
  y17 = np.array([391, 173, 1.])
  x18 = np.array([550, 170, 1.])
  y18 = np.array([489, 211, 1.])
  x19 = np.array([511, 93, 1.])
  y19 = np.array([529, 147, 1.])
  x20 = np.array([413, 120, 1.])
  y20 = np.array([433, 110, 1.])


  # Vektori tih tacaka
  xx = np.array([x5, x6, x9, x10, x17, x18, x19, x20])
  yy = np.array([y5, y6, y9, y10, y17, y18, y19, y20])

  # Normalizacija tacaka
  tx, xxx = normalizuj(xx)
  ty, yyy = normalizuj(yy)

  # Jednacina y^T * F * x = 0, gde su nepoznate
  # koeficijenti trazene fundamentalne matrice
  jed = lambda x, y: np.array([np.outer(y, x).flatten()])


  # Matrica formata 8x9 koja predstavlja osam
  # jednacina dobijenih iz korespondencija

  jed8 = np.concatenate([jed(x, y) for x, y in zip(xxx, yyy)])

  # DLT algoritam, SVD dekompozicija
  SVDJed8 = LA.svd(jed8)

  # Vektor koeficijenata fundamentalne matrice,
  # dobijen kao poslednja vrsta matrice V^T
  Fvector = SVDJed8[-1][-1]

  # Fundamentalna matrica napravljena od vektora
  FFt = Fvector.reshape(3, 3)

  #posizanje uslova da je determinanta bliska 0

  # SVD dekompozicija fundamentalne matrice
  Ut, DDt, VTt = LA.svd(FFt)

  # Zeljena matrica je singularna
  DD1t = np.diag([1, 1, 0]) @ DDt
  DD1t = np.diag(DD1t)

  # Zamena matrice DD novom DD1 i tako
  # dobijanje nove fundamentalne matrice
  FF1t = Ut @ DD1t @ VTt

  # Vracanje u pocetni koordinatni sistem
  FF = ty.T @ FF1t @ tx

 #ODREDJIVANJE EPIPOLOVA

  # SVD dekompozicija fundamentalne matrice
  U, _, VT = LA.svd(FF)

  # Treca vrsta V^T je prvi epipol, resenje jednacine
  # F * e1 = 0, najmanja sopstvena vrednost matrice
  e1 = VT[-1]

  # Afine koordinate prvog epipola

  e1 = normalizujJednu(e1)

  # Za drugi epipol ne resavamo F^T * e2 = 0,
  # vec primecujemo da je SVD dekompozicija F^T
  # transponat one od F, tako da je drugi epipol
  # poslednja kolona matrice U prvog razlaganja
  e2 = U[:, -1]

  # Afine koordinate drugog epipola
  e2 = normalizujJednu(e2)

  #REKONSTRUKCIJA SKRIVENIH TACAKA

  # Preostale vidljive tacke
  x1  = np.array([341, 577, 1.])
  y1  = np.array([165, 510, 1.])
  x2  = np.array([338, 453, 1.])
  y2  = np.array([263, 424, 1.])
  x3  = np.array([213, 519, 1.])
  y3  = np.array([125, 402, 1.])
  x4  = np.array([296, 457, 1.])
  y4 = np.array([186, 366, 1.])
  x8 = np.array([198, 383, 1.])
  y8 = np.array([229, 309, 1.])
  x11 = np.array([287, 30, 1.])
  y11 = np.array([467, 37, 1.])
  x12 = np.array([174, 235, 1.])
  y12 = np.array([226, 110, 1.])
  x13 = np.array([444, 258, 1.])
  y13 = np.array([383, 252, 1.])
  x14 = np.array([545, 229, 1.])
  y14 = np.array([476, 287, 1.])
  x16 = np.array([409, 180, 1.])
  y15 = np.array([517, 226, 1.])

  # Neophodno je naci koordinate nevidljivih tacaka

  # Nevidljive tacke prve projekcije
  Xb = np.cross(np.cross(x5, x9), np.cross(x6, x10))
  Yb = np.cross(np.cross(x10, x9), np.cross(x11, x12))

  x7 = np.cross(np.cross(x11, Xb), np.cross(x8, Yb))
  x7 = (x7/x7[-1]).round()

  Xb = np.cross(np.cross(x13, x17), np.cross(x14, x18))
  Yb = np.cross(np.cross(x18, x17), np.cross(x19, x20))

  x15 = np.cross(np.cross(x19, Xb), np.cross(x16, Yb))
  x15 = (x15/x15[-1]).round()


  # Nevidljive tacke druge projekcije
  
  Xb = np.cross(np.cross(y5, y9), np.cross(y8, y12))
  Yb = np.cross(np.cross(y12, y9), np.cross(y8, y5))

  y7 = np.cross(np.cross(y11, Xb), np.cross(y6, Yb))
  y7 = (y7/y7[-1]).round()  

  Xb = np.cross(np.cross(y14, y18), np.cross(y15, y19))
  Yb = np.cross(np.cross(y19, y18), np.cross(y20, y17))

  y16 = np.cross(np.cross(y20, Xb), np.cross(y13, Yb))
  y16 = (y16/y16[-1]).round()


  # TRIANGULACIJA

  # Kanonska matrica kamere
  T1 = np.hstack([np.eye(3), np.zeros(3).reshape(3, 1)])

  # Matrica vektorskog mnozenja
  vec1=lambda p: np.array([[  0,   -p[2],  p[1]],
                            [ p[2],   0,   -p[0]],
                            [-p[1],  p[0],   0  ]])

  # Matrica drugog epipola
  E2 = px(e2)


  # Druga matrica kamere
  T2 = np.hstack([E2 @ FF, e2.reshape(3, 1)])

  # Za svaku tacku po sistem od cetiri jednacine
  # sa cetiri homogene nepoznate, mada mogu i tri
  jednacine = lambda xx, yy: np.array([ xx[1]*T1[2] - xx[2]*T1[1],
                                       -xx[0]*T1[2] + xx[2]*T1[0],
                                        yy[1]*T2[2] - yy[2]*T2[1],
                                       -yy[0]*T2[2] + yy[2]*T2[0]])


  # Fja koja vraca 3D koordinate rekonstruisane tacke
  TriD = lambda xx, yy: UAfine(LA.svd(jednacine(xx, yy))[-1][-1])

  # Piksel koordinate sa obe slike
  slika1 = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9,
                     x10, x11, x12, x13, x14, x15, x16,
                   x17, x18, x19, x20])
  
  slika2 = np.array([y1, y2, y3, y4, y5, y6, y7, y8, y9,
                     y10, y11, y12, y13, y14, y15, y16,
                   y17, y18, y19, y20])

  # Rekonstruisane 3D koordinate tacaka
  rekonstruisane = np.array([TriD(x, y) for x, y
                             in zip(slika1, slika2)])

  # Skaliranje z-koordinate u cilju poboljsanja rezultata

  rekonstruisane380=[]
  for i in range(len(rekonstruisane)):
    rekonstruisane380.append(np.diag([1,1,380])@rekonstruisane[i])
  
  # Ivice rekonstruisanih objekata
  iviceMala = np.array([[1, 2], [2, 3], [3, 1], [4, 1],
                        [4,2], [4,3]])

  iviceSrednja = np.array([[ 5,6], [6,7], [7,8], [8,5],
                           [9,10], [10,11], [11,12], [12,9],
                           [ 12,8], [9,5], [10,6], [11,7]])

  iviceVelika = np.array([[13,14], [14,15], [15,16], [16,13],
                          [17,18], [18,19], [19,20], [20, 17],
                          [17, 13], [18, 14], [19, 15], [20, 16]])

  # Vracanje rezultata
  return rekonstruisane380, iviceMala, iviceSrednja, iviceVelika



# Enumeracija bafera tipki
PRAZNO = 0      # sve nule
LEVO = 16       # 1 << 4
DESNO = 32      # 1 << 5
RESET = 64      # 1 << 6
tipke = PRAZNO  # prazan


# Klasa za predstavljanje oka/kamere
class Oko:
  # Konstruktor oka
  def __init__(self, x, y, z, cx, cy, cz):
    # Greska u slucaju zakljucavanja
    if x == cx and y == cy:
      raise ValueError
    
    # Inicalizacija Dekartovih koordinata
    self.x = x
    self.y = y
    self.z = z

    # Inicijalizacija sfernih koordinata
    self.r = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
    self.phi = np.arctan2(y-cy, x-cx)
    self.theta = np.arcsin((z-cz)/self.r)

    # Cuvanje pocetnih vrednosti
    self.r0 = self.r
    self.phi0 = self.phi
    self.theta0 = self.theta
    
    # Inicijalizacija sfernog pomeraja
    self.d_r = 10
    self.d_phi = np.pi/100
    self.d_theta = np.pi/100

    # Inicijalizacija sfernog minimuma
    self.r_min = 1000
    self.phi_min = -np.pi
    self.theta_min = -np.pi/2 + np.pi/20

    # Inicijalizacija sfernog maksimuma
    self.r_max = 3000
    self.phi_max = np.pi
    self.theta_max = np.pi/2 - np.pi/20
    
    # Inicijalizacija centra pogleda
    self.cx = cx
    self.cy = cy
    self.cz = cz

  # Izracunavanje Dekartovih koordinata
  def popravi(self):
    self.x = self.cx + self.r * np.cos(self.theta) * np.cos(self.phi)
    self.y = self.cy + self.r * np.cos(self.theta) * np.sin(self.phi)
    self.z = self.cz + self.r * np.sin(self.theta)

  
  # Oko ide nalevo
  def levo(self):
    # Svako kretanje kamere
    # zaustavlja resetovanje
    global tipke
    tipke &= ~RESET
    
    # Smanjuje se azimut
    self.phi -= self.d_phi
    
    # Popravka jer phi = [-pi, pi)
    if self.phi < self.phi_min:
        self.phi += 2*np.pi

  # Oko ide nadesno
  def desno(self):
    # Svako kretanje kamere
    # zaustavlja resetovanje
    global tipke
    tipke &= ~RESET
    
    # Povecava se azimut
    self.phi += self.d_phi
    
    # Popravka jer phi = [-pi, pi)
    if self.phi >= self.phi_max:
        self.phi -= 2*np.pi

  # Vracanje oka na pocetni polozaj
  def reset(self):
    # Kamera se u sfernom koordinatnom sistemu
    # posmatra kao trojka (r, phi, theta); cilj
    # resetovanja je da tu trojku transformise
    # u (r0, phi0, theta0), gde su nulom indeksirane
    # pocetne vrednosti parametara; ova tranformacija
    # moze biti translacija po vektoru pomeraja
    t_r = self.r0 - self.r
    t_phi = self.phi0 - self.phi
    t_theta = self.theta0 - self.theta
    
    # Duzina prvog vektora
    duzina1 = np.abs(t_r)/10
    kraj1 = True

    # Normalizacija izracunatog vektora
    # deljenjem sa njegovom duzinom
    if duzina1 != 0:
      t_r /= duzina1

      # Translacija po radijusu pogleda,
      # ali samo ako je vrednost pomeraja
      # manja od udaljenosti bitnih tacaka
      if abs(t_r) < abs(self.r0 - self.r):
        self.r += t_r
        kraj1 = False
      else:
        self.r = self.r0

    # Isti slucaj je i sa drugim vektorom
    duzina2 = 50*np.sqrt(t_phi**2 + t_theta**2)
    kraj2 = True
    if duzina2 != 0:
      t_phi /= duzina2
      t_theta /= duzina2
    
      # Slicno kao dosad, samo za azimutalni ugao
      if abs(t_phi) < abs(self.phi0 - self.phi):
        self.phi += t_phi
        kraj2 = False
      else:
        self.phi = self.phi0
      
      # Slicno kao dosad, samo za polarni ugao
      if abs(t_theta) < abs(self.theta0 - self.theta):
        self.theta += t_theta
        kraj2 = False
      else:
        self.theta = self.theta0
    
    # Indikator da li je vracanje zavrseno
    return kraj1 and kraj2

# Globalni identifikator tajmera
TIMER_ID = 0

# Globalno vreme osvezavanja tajmera
TIMER_INTERVAL = 20

# Dimenzije prozora
sirina, visina = 600, 600

# Parametri rekonstrukcije
rek, ivm, ivs, ivv = rekonstruisi()
c = np.mean(rek, axis = 0)

# Globalno oko/kamera
oko = Oko(c[0]+1010, c[1]-1170, c[2]-745,
             c[0],      c[1],     c[2])



# Funkcija koja se poziva na tajmer
def tajmer(*args):
  # Uzimanje tipki
  global tipke

  # Resetovanje pogleda
  if tipke & RESET:
    # Kraj animacije resetovanja
    if oko.reset():
      tipke &= ~RESET
  
  # Oko ide nalevo
  if tipke & LEVO:
    oko.levo()

  # Oko ide nadesno
  if tipke & DESNO:
    oko.desno()

  # Forsiranje ponovnog iscrtavanja
  glutPostRedisplay()

  # Ponovno postavljanje tajmera
  glutTimerFunc(TIMER_INTERVAL, tajmer, TIMER_ID)

# Funkcija za obradu otpustanja tipki
def tipkeg(taster, *args):
  global tipke
  if taster in [b'a', b'A']:
    tipke &= ~LEVO
  elif taster in [b'd', b'D']:
    tipke &= ~DESNO
  

# Funkcija za obradu dogadjaja tastature
def tipked(taster, *args):
  # Uzimanje tipki
  global tipke

  # Prekid programa u slucaju Esc,
  # sto je 1b u hex sistemu zapisa
  if taster == b'\x1b':
    sys.exit()
  # Obrada tipki za kretanje kamere
  elif taster in [b'a', b'A']:
    tipke |= LEVO
  elif taster in [b'd', b'D']:
    tipke |= DESNO
  elif taster in [b'r', b'R']:
    tipke ^= RESET

# Crtanje rekonstruisanog objekta
def objekat():
  # Pocetak iscrtavanja linija
  glBegin(GL_LINES)

  # Ljubicasta boja za gornju kutiju
  glColor3f(1, 0, 1)

  # Crtanje svake ivice
  for i, j in ivm:
    glVertex3f(*rek[i-1])
    glVertex3f(*rek[j-1])

  # Plava boja za srednju kutiju
  glColor3f(0, 0, 1)

  # Crtanje svake ivice
  for i, j in ivs:
    glVertex3f(*rek[i-1])
    glVertex3f(*rek[j-1])

  # Crvena boja za donju kutiju
  glColor3f(1, 0, 0)

  # Crtanje svake ivice
  for i, j in ivv:
    glVertex3f(*rek[i-1])
    glVertex3f(*rek[j-1])

  # Kraj iscrtavanja linija
  glEnd()
    
# Funkcija za prikaz scene
def prikaz():
  # Ciscenje prozora: ciscenje bafera boje i dubine;
  # prothodni sadrzaj prozora brise se tako sto se boja
  # svih piksela postavlja na zadatu boju pozadine
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

  # Postavljanje matrice transformacije;
  # ucitava se jedinicna matrica koja se
  # kasnije mnozi matricama odgovarajucih
  # geometrijskih transformacija
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()

  # Izracunavanje polozaja oka
  oko.popravi()

  # Vidni parametri; scena se tranformise tako
  # da se oko nadje ispred, a objekti na centru
  # scene, cime se simulira sinteticka kamera 
  gluLookAt( oko.x,  oko.y,  oko.z,  # Polozaj oka/kamere
            oko.cx, oko.cy, oko.cz,  # Srediste pogleda
               0,      0,      1   ) # Vektor normale

  # Crtanje rekonstruisanog objekta
  objekat()

  # Zamena iscrtanih bafera
  glutSwapBuffers()

# Glavna (main) fja
def main():
  glutInit();
  # Opis: RGB (crvena, zelena, plava) sistem
  # boja, bafer dubine za pravilno postavljanje
  # objekata, dva bafera scene zarad manjeg
  # seckanja prilikom postavljanja nove slike
  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE)

  # Pravljenje prozora: usput ide
  # postavljanje dimenzija i imena
  glutInitWindowSize(sirina, visina)
  glutCreateWindow(b'3D rekonstrukcija')

  # Postavljanje sive za boju 'ciscenja prozora',
  # koja se ujedno uzima za zadatu boju pozadine
  glClearColor(0.96, 0.96, 0.96, 1)

  # Ukljucivanje umeksavanja linija
  glEnable(GL_LINE_SMOOTH)

  # Postavljanje sirine linije
  glLineWidth(4)

  # Ukljucivanje provere dubine
  glEnable(GL_DEPTH_TEST)

  # Vezivanje funkcija za prikaz,
  # mis, tastaturu i promenu prozora
  glutDisplayFunc(prikaz)
  glutKeyboardFunc(tipked)
  glutKeyboardUpFunc(tipkeg)
                       # Sprecava se promena dimenzija
  glutReshapeFunc(lambda *args: glutReshapeWindow(sirina, visina))

  # Postavljanje glavnog tajmera
  glutTimerFunc(TIMER_INTERVAL, tajmer, TIMER_ID)

  # Postavljanje matrice projekcije
  glMatrixMode(GL_PROJECTION)
  gluPerspective(40,            # Ugao perspektive
                 sirina/visina, # Odnos dimenzija prozora
                 1,             # Prednja ravan odsecanja
                 5000)          # Zadnja ravan odsecanja

  # Glavna petlja animacije
  glutMainLoop()

main();