import numpy as np


class Gradient_v3:
	@staticmethod
	def calculate_grad(M, U, S, V):
		urows = np.size(U, 0)
		ucolumns = np.size(U, 1)
		srows = np.size(S, 0)
		scolumns = np.size(S, 1)
		vrows = np.size(V, 0)
		vcolumns = np.size(V, 1)
		gradU = np.zeros((urows, ucolumns))
		gradS = np.zeros((srows, scolumns))
		gradV = np.zeros((vrows, vcolumns))
		for i in range(urows):
			for j in range(ucolumns):
				sum1 = 0
				for k in range(vrows):
					sum2 = 0
					for l in range(srows):
						sum2+=U.item((i,l))*S.item((l,l))*V.item((k,l))
					sum1+=V.item((k,j))*(M.item((i,k))-sum2)
				gradU.itemset((i,j), (-2)*S.item((j,j))*sum2)
		for i in range(srows):
			sum1 = 0
			for j in range(urows):
				sum2 = 0
				for k in range(vrows):
					sum3 = 0
					for l in range(srows):
						sum3 += U.item((j,l))*S.item((l,l))*V.item((k,l))
					sum2+=2*(M.item((j,k)) - sum3)*((-1)*U.item((j,i))*V.item((k,i)))
				sum1 += sum2
			gradS.itemset((i,i),sum1)
		for i in range(vrows):
			for j in range(vcolumns):
				sum1 = 0
				for k in range(urows):
					sum2 = 0
					for l in range(srows):
						sum2 += U.item((k,l))*S.item((l,l))*V.item((i,l))
					sum1+=U.item((k,j))*(M.item((k,i)) - sum2)
				gradV.itemset((i,j),(-2)*S.item((j,j))*sum1)
		return gradU, gradS, gradV

	@staticmethod
	def initialize_USV(M):
		xmax=abs(max(M.min(), M.max(), key=abs))
		n = np.size(M,0)
		m = np.size(M,1)
		r = min(n, m)
		U = np.random.rand(n, r)*2-1
		S = np.zeros((r,r))
		V = np.random.rand(m, r)*2-1
		for i in range(r):
			S.itemset((i,i),np.random.rand()*xmax)
		return U, S, V

	@staticmethod
	def normalize(A):
		norms = []
		rows = np.size(A,0)
		columns = np.size(A,1)
		normalized_A = np.zeros((rows,columns))
		for i in range(columns):
			a = A[:,i]
			norm = np.linalg.norm(a)
			a = a/norm
			for j in range(rows):
				normalized_A.itemset((j,i), a[j])
			norms.append(norm)
		return norms, normalized_A

	@staticmethod
	def normalize_USV(U,S,V):
		normsU, Unew = Gradient_v3.normalize(U)
		size = np.size(S,0)
		normsV, Vnew = Gradient_v3.normalize(V)
		Snew = np.zeros((size,size))
		for i in range(size):
			Snew.itemset((i,i), S.item((i,i))+normsU[i]+normsV[i])
		return Unew, Snew, Vnew

	@staticmethod
	def calculate_svd(M):
		U, S, V = Gradient_v3.initialize_USV(M)
		a = 0.1
		for i in range(100):
			gradU, gradS, gradV = Gradient_v3.calculate_grad(M, U, S, V)
			U = U - a*gradU
			S = S - a*gradS
			V = V - a*gradV
			U, S, V = Gradient_v3.normalize_USV(U, S, V)
		return U, S, V