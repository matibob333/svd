import numpy as np


class Gradient_v4:
	@staticmethod
	def calculate_gradient_highest_sigma(M, v, number_of_var):
		grad = np.zeros((np.size(v, 0), 1))
		for i in range(number_of_var):
			item = 0
			for j in range(np.size(M, 0)):
				inner_sum = 0
				for k in range(np.size(M, 1)):
					inner_sum += M.item((j, k))*v.item((k))
				inner_sum = M.item((j, i))*inner_sum
				item += inner_sum
			item = item*(-2)
			grad.itemset((i, 0), item)
		return grad


	@staticmethod
	def calculate_last_parameters(V, number_of_vars, vec):
		A = (V[0:number_of_vars,:]).T
		x = vec[0:number_of_vars,:]
		b = -(A.dot(x))
		B = (V[number_of_vars:,:]).T
		C = np.copy(B)
		for i in range(np.size(C,0)):
			for j in range(np.size(C,1)):
				if(C.item((i,j))==0):
					C.itemset((i,j),1e-6)
		y = np.linalg.solve(C,b)
		ret = np.copy(vec)
		for i in range(number_of_vars,np.size(ret,0)):
			ret.itemset((i,0),y.item((i-number_of_vars,0)))
		return ret

	@staticmethod
	def find_next_vector(M, V, number_of_vars):
		#vec = V[:,np.size(V,1)-1:np.size(V,1)]
		vec = np.ones((np.size(V,0), 1))
		norm = np.linalg.norm(vec)
		vec = Gradient_v4.calculate_last_parameters(V, number_of_vars,vec)
		norm = np.linalg.norm(vec)
		vec = np.divide(vec,norm)
		a = 0.5
		for i in range(1000):
			grad = Gradient_v4.calculate_gradient_highest_sigma(M,vec,number_of_vars)
			vec = np.subtract(vec,np.multiply(grad,a))
			vec = Gradient_v4.calculate_last_parameters(V, number_of_vars, vec)
			norm = np.linalg.norm(vec)
			vec = np.divide(vec,norm)
		return vec

	@staticmethod
	def find_highest_sigma_vector(M):
		#utworzenie losowego wektora jednostkowego
		v = np.random.rand(np.size(M, 1), 1)*2-1
		v = np.divide(v, np.linalg.norm(v))
		a = 0.5
		for i in range(1000):
			grad = Gradient_v4.calculate_gradient_highest_sigma(M, v,np.size(M,1))
			v = np.subtract(v,np.multiply(grad, a))
			v = np.divide(v, np.linalg.norm(v))
		return v

	@staticmethod
	def calculate_svd(M, param = -1):
		if param == -1:
			param = min(np.size(M, 0), np.size(M, 1))
		C = np.copy(M)
		for i in range(np.size(C,0)):
			for j in range(np.size(C,1)):
				if(C.item((i,j))==0):
					C.itemset((i,j),1e-12)
		V = Gradient_v4.find_highest_sigma_vector(C)
		U = C.dot(V)
		S = []
		s = np.linalg.norm(U)
		U = np.divide(U,s)
		S.append(s)
		for i in range(1, param):
			v = Gradient_v4.find_next_vector(C,V,np.size(V,0)-i)
			u = C.dot(v)
			s = np.linalg.norm(u)
			if s >= 1e-5 or s <=-1e-5:
				u = np.divide(u,s)
			else:
				u = np.ones((np.size(U,0), 1))
				norm = np.linalg.norm(u)
				u = np.divide(u,norm)
				u = Gradient_v4.calculate_last_parameters(U,np.size(U,0)-i,u)
				norm = np.linalg.norm(u)
				u = np.divide(u,norm)
			V = np.c_[V, v]
			U = np.c_[U, u]
			S.append(s)
		Sdiag = np.zeros((len(S),len(S)))
		for i in range(len(S)):
			Sdiag.itemset((i,i),S[i])
		return U, Sdiag, V