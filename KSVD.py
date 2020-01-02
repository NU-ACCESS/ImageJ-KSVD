from ij import IJ, ImagePlus
from ij.plugin import ImageCalculator
from ij.process import ImageProcessor
from ij.process import FloatProcessor
from mpv2 import MatchingPursuit as MP, JamaMatrix as Matrix, SymmetricMatrix as SM
from org.ejml.simple import SimpleMatrix 
from org.ejml.simple import SimpleSVD 
import cmath, math
import random

def find(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

noIt_RS = 6

noIt_DL = 6

Sparsity = 4

#get initial dictionary D and construct matrix
IJ.run("Text Image... ", "open=/Users/homelyp/Desktop/dicinaround.txt")#path to initial dictionary
imp = IJ.getImage()
m = imp.getProcessor().getPixels()
m2 = [val for val in m]
DicSize = imp.width
Dinit = [m2[i:i+DicSize] for i in range(0, len(m2), DicSize)]
imp.close()

#get signal X and construct matrix
imp2 = IJ.getImage()
n_slices = imp2.getStack().getSize()
X =[]
for i in range(1, n_slices+1):
  imp2.setSlice(i) 
  n = imp2.getProcessor().getPixels()   
  n2 = [val for val in n]
  X.append(n2)
x = zip(*X)#transpose signal X


#iterates through random selection of 5% of pixels N times
for i in range(0,noIt_RS):
	if i == 0:
		D = Dinit
	
	percent = int((imp2.height*imp2.width)*0.05)

	subX =[]

	for i in range(0,percent):
		rand = random.choice(x)
		subX.append(rand)

	p = len(subX)# numb of pixels in image
	
#iterates through K-SVD J times
	for j in range(0,noIt_DL):
		
		jD = Matrix(D)
		jDD = SM(DicSize,DicSize).eqInnerProductMatrix(jD)
		jMP = MP(jD,jDD)

		# orthogonal matching pursuit to create a sparse appoximation
		w =[]
		for i in range(0, p):
			q = subX[i]
			W = jMP.vsOMP(q,Sparsity)  
			w.append(W)


		Sp = Matrix(w).getColumnPackedCopy()
		m3 = [val for val in Sp]
		w = [m3[i:i+percent] for i in range(0, len(m3), percent)]

		X = SimpleMatrix(subX).transpose()
		X.printDimensions()
		W = SimpleMatrix(w)
		D1 = SimpleMatrix(D)# Dictionary


		R = X.minus(D1.mult(W))

		K=DicSize # numb of atoms

		DD =[]

		for k in range(0,K):

			I = find([[val for val in W.getMatrix().data][i:i+len(w[0])] for i in range(0, len(w[0])*len(w), len(w[0]))][k], lambda x: x > 0)

			Ri =[]
			for i in range(0,len(I)):
				h = I[i]
				ri = R.extractVector(0,h).getMatrix().data 
				Ri.append(ri)
	
			Di =[]
			for i in range(0,len(I)):
				h = I[i]
				ri = W.get(k,h)
				Di.append(ri)

			Ri2= SimpleMatrix(Ri).transpose().plus((D1.extractVector(0,k)).mult(SimpleMatrix([Di])))
	 		D_new = Ri2.svd().getU().extractVector(0,0).getMatrix().data
	 		DD.append(D_new)
 	
	 	D = zip(*DD)

#Reconstructed = (D1.mult(W)).getMatrix().data

d2 = [y for x in D for y in x]
Rec = ImagePlus("Denoised_Dictionary", FloatProcessor(DicSize,imp.height,d2))

Rec.show()
