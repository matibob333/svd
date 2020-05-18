import numpy as np


class Specimen:
	def __init__(self, U, S, Vt):
		self.U = U
		self.S = S
		self.Vt = Vt

	def calculate_product(self):
		return np.dot(np.dot(self.U, self.S), self.Vt)