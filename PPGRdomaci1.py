from matplotlib import pyplot as plt 
from matplotlib import image as mpimg
image=mpimg.imread("leva_oznake2022(1).jpg")

def crossProduct(first, second):
	return (first[1]*second[2]-first[2]*second[1], (-1)*(first[0]*second[2]-first[2]*second[0]), first[0]*second[1]-first[1]*second[0])

def prebaciIzAfinih(dots):
	newList=[]
	for dot in dots:
		newDot=[dot[0], dot[1], 1]
		newList.append(newDot)
	return newList
	
def UAfine(dot):
	return [round(dot[0]/dot[2]), round(dot[1]/dot[2]),1]

def lepIspis(dot):
	return [round(dot[0]), round(dot[1]),1]
	
def average(x1,x2,x3):
	return ((x1[0]+x2[0]+x3[0])/3, (x1[1]+x2[1]+x3[1])/3, (x1[2]+x2[2]+x3[2])/3)

fig, ax=plt.subplots()
ax.imshow(image)

print("Unesite sedam tacaka misem: ")

while(True):
	yroi=plt.ginput()
newList=prebaciIzAfinih(yroi)

i=1
print("Uneli ste sledece koordinate:")
for item in newList:
	if i<4:
		print("Dot ", i, ": ", lepIspis(item))
	else:
		print("Dot ", i+1, ": ", lepIspis(item)) 
	i=i+1
#newList=[[595,301,1], [292,517,1], [157,379,1], [665,116,1], [304,295,1], [135,163,1], [509,43,1]]

def missing(dots):
	x1_inf=UAfine(crossProduct(crossProduct(newList[1], newList[4]), crossProduct(newList[0], newList[3])))
	x2_inf=UAfine(crossProduct(crossProduct(newList[1], newList[4]), crossProduct(newList[2], newList[5])))
	x3_inf=UAfine(crossProduct(crossProduct(newList[0], newList[3]), crossProduct(newList[2], newList[5])))
	x_inf=UAfine(average(x1_inf,x2_inf,x3_inf))
	y_inf=crossProduct(crossProduct(newList[3], newList[4]), crossProduct(newList[5], newList[6]))
	return UAfine(crossProduct(crossProduct(newList[6], x_inf), crossProduct(newList[2], y_inf)))
print("Koordinate tacke koja nedostaje su:", missing(newList))
plt.show()


