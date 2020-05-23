import numpy as np

class Gradient_v2:
	@staticmethod
	def calculate_gradient_highest_sigma(M, v):
		grad = np.zeros((np.size(v, 0), 1))
		for i in range(np.size(v, 0)):
			item = 0
			for j in range(np.size(M, 0)):
				inner_sum = 0
				for k in range(np.size(M, 1)):
					inner_sum += M.item((j, k))*v.item((k, 0))
				inner_sum = M.item((j, i))*inner_sum
				item += inner_sum
			item = item*(-2)
			grad.itemset((i, 0), item)
		return grad

	@staticmethod
	def find_highest_sigma_vector(M):
		#utworzenie losowego wektora jednostkowego
		v = np.random.rand(np.size(M, 1), 1)*2-1
		v = np.divide(v, np.linalg.norm(v))
		a = 0.5
		for i in range(1000):
			grad = Gradient_v2.calculate_gradient_highest_sigma(M, v)
			v = np.subtract(v,np.multiply(grad, a))
			v = np.divide(v, np.linalg.norm(v))
		return v

	@staticmethod
	def find_orthogonal_vector(V):
		v = np.random.rand(np.size(V, 0), 1)*2-1
		v = np.divide(v, np.linalg.norm(v))
		a = 0.05
		for i in range(1000):
			grad = Gradient_v2.calculate_gradient_highest_sigma(V.T, v)
			grad = grad * (-1.0)
			v = np.subtract(v,np.multiply(grad, a))
			v = np.divide(v, np.linalg.norm(v))
		return v

	@staticmethod
	def calculate_svd(M):
		V = Gradient_v2.find_highest_sigma_vector(M)
		U = M.dot(V)
		S = []
		S.append(np.linalg.norm(U))
		U = U/S[0]
		for i in range(1, min(np.size(M, 0), np.size(M, 1))):
			for j in range(100):
				powtorzenie = False
				v = Gradient_v2.find_orthogonal_vector(V)
				u = M.dot(v)
				s = np.linalg.norm(u)
				for k in range(1, np.size(S)):
					s_test = S[k]-s
					if(np.linalg.norm(s_test)<0.001):
						powtorzenie = True
						break
				if(powtorzenie==False):
					u = u/s
					V = np.c_[V, v]
					U = np.c_[U, u]
					S.append(s)
					break
		return U, S, V