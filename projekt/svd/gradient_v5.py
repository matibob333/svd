import numpy as np


class Gradient_v5:
	@staticmethod
	def calculate_gradient_highest_sigma(M, v, number_of_var, indexes):
		grad = np.zeros((np.size(v, 0), 1))
		for i in indexes:
		#for i in range(number_of_var):
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
	def get_vector_of_indexes(V,number_of_vars):
		indexes_to_b = [];
		num_of_V_rows = np.size(V,0)
		num_of_V_columns = np.size(V,1)
		minInRows = np.amin(np.absolute(V), axis=1)
		minInRowsSorted = np.sort(minInRows)
		for i in range(number_of_vars):
			it=0
			for j in minInRows:
				if j==minInRowsSorted[i]:
					indexes_to_b.append(it)
					break
				it+=1
		return indexes_to_b


	@staticmethod
	def calculate_last_parameters(V, number_of_vars, vec, indexes_to_b):
		A = np.zeros((0,np.size(V,1)))
		x = np.zeros((0,1))
		#for i in range(np.size(V,0)):
		#	if i in indexes_to_b:
		#		A = np.r_[A,V[i:i+1,:]]
		#		x = np.r_[x,V[i:i+1,:]]
				

		for i in range(len(indexes_to_b)):
			A = np.r_[A,V[indexes_to_b[i]:indexes_to_b[i]+1,:]]
			x = np.r_[x,vec[indexes_to_b[i]:indexes_to_b[i]+1,:]]
		A = A.T

		#A = (V[0:number_of_vars,:]).T
		#x = vec[0:number_of_vars,:]
		b = -(A.dot(x))

		B = np.zeros((0,np.size(V,1)))
		for i in range(np.size(V,0)):
			if i not in indexes_to_b:
				B = np.r_[B,V[i:i+1,:]]
		B = B.T

		#B = (V[number_of_vars:,:]).T
		y = np.linalg.solve(B,b)
		ret = np.copy(vec)

		j = 0
		for i in range(np.size(V,0)):
			if i not in indexes_to_b:
				ret.itemset((i,0),y.item((j,0)))
				j+=1

		#for i in range(number_of_vars,np.size(ret,0)):
		#	ret.itemset((i,0),y.item((i-number_of_vars,0)))
		return ret

	@staticmethod
	def find_next_vector(M, V, number_of_vars):
		#vec = V[:,np.size(V,1)-1:np.size(V,1)]
		vec = np.ones((np.size(V,0), 1))
		norm = np.linalg.norm(vec)
		vec = np.divide(vec,norm)
		indexes = Gradient_v5.get_vector_of_indexes(V, number_of_vars)
		vec = Gradient_v5.calculate_last_parameters(V, number_of_vars,vec,indexes)
		norm = np.linalg.norm(vec)
		vec = np.divide(vec,norm)
		a = 0.5
		for i in range(100):
			grad = Gradient_v5.calculate_gradient_highest_sigma(M,vec,number_of_vars, indexes)
			vec = np.subtract(vec,np.multiply(grad,a))
			vec = Gradient_v5.calculate_last_parameters(V, number_of_vars, vec, indexes)
			norm = np.linalg.norm(vec)
			vec = np.divide(vec,norm)
		return vec

	@staticmethod
	def find_highest_sigma_vector(M):
		#utworzenie losowego wektora jednostkowego
		v = np.random.rand(np.size(M, 1), 1)*2-1
		v = np.divide(v, np.linalg.norm(v))
		a = 0.5
		for i in range(100):
			grad = Gradient_v5.calculate_gradient_highest_sigma(M, v,np.size(M,1), range(np.size(M,1)))
			v = np.subtract(v,np.multiply(grad, a))
			v = np.divide(v, np.linalg.norm(v))
		return v

	@staticmethod
	def calculate_svd(M, param = -1):
		if param == -1:
			param = min(np.size(M, 0), np.size(M, 1))
		
		V = Gradient_v5.find_highest_sigma_vector(M)
		U = M.dot(V)
		S = []
		s = np.linalg.norm(U)
		U = np.divide(U,s)
		S.append(s)
		for i in range(1, param):
			v = Gradient_v5.find_next_vector(M,V,np.size(V,0)-i)
			u = M.dot(v)
			s = np.linalg.norm(u)
			u = np.divide(u,s)
			V = np.c_[V, v]
			U = np.c_[U, u]
			S.append(s)
		Sdiag = np.zeros((len(S),len(S)))
		for i in range(len(S)):
			Sdiag.itemset((i,i),S[i])
		return U, Sdiag, V