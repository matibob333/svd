import numpy as np
import copy
import scipy
from scipy.stats import ortho_group
import scipy.linalg as lin
from specimen import Specimen

#Algorytm genetyczny rozszerzony dla tworzenia wszystkich macierzy U, Sigma i V
class Genetic_extended:
	@staticmethod
	def create_population(M):
		xmax=abs(max(M.min(), M.max(), key=abs))
		rows=np.size(M,0)
		columns=np.size(M,1)
		population = []
		for i in range(20):
			S=np.zeros((rows,columns))
			for j in range(min(rows,columns)):
				S.itemset((j,j),np.random.rand()*xmax)
			U = ortho_group.rvs(rows)
			Vt = ortho_group.rvs(columns)
			item = Specimen(U, S, Vt)
			population.append(item)
		return population

	@staticmethod
	def crossover(ParentA, ParentB):
		if np.random.rand() < 0.8:
			weight = np.random.rand()
			n = 7
			int_weight = np.random.randint(7)
			rows = np.size(ParentA.S, 0)
			columns = np.size(ParentA.S, 1)
			#liczenie pot�g macierzy U do �redniej geometrycznej wa�onej
			Ua1_weighted = np.linalg.matrix_power(ParentA.U, int_weight)
			Ub1_weighted = np.linalg.matrix_power(ParentB.U, n-int_weight)
			Ua2_weighted = np.linalg.matrix_power(ParentA.U, n-int_weight)
			Ub2_weighted = np.linalg.matrix_power(ParentB.U, int_weight)
			#liczenie pot�g macierzy Vt do �redniej geometrycznej wa�onej
			Vta1_weighted = np.linalg.matrix_power(ParentA.Vt, int_weight)
			Vtb1_weighted = np.linalg.matrix_power(ParentB.Vt, n-int_weight)
			Vta2_weighted = np.linalg.matrix_power(ParentA.Vt, n-int_weight)
			Vtb2_weighted = np.linalg.matrix_power(ParentB.Vt, int_weight)
			#utworzenie dzieci ze �rednich wa�onych i pocz�tkowo macierzy zerowej jako Sigma (do dalszego uzupe�niania)
			ChildA = Specimen(np.real(lin.fractional_matrix_power(Ua1_weighted.dot(Ub1_weighted), 1/n)),np.zeros((rows, columns)),np.real(lin.fractional_matrix_power(Vta1_weighted.dot(Vtb1_weighted), 1/n)))
			ChildB = Specimen(np.real(lin.fractional_matrix_power(Ua2_weighted.dot(Ub2_weighted), 1/n)),np.zeros((rows, columns)),np.real(lin.fractional_matrix_power(Vta2_weighted.dot(Vtb2_weighted), 1/n)))
			#uzupe�nianie sigm
			for i in range(min(rows, columns)):
				ChildA.S.itemset((i,i), weight*ParentA.S.item((i,i)) + (1-weight)*ParentB.S.item((i,i)))
				ChildB.S.itemset((i,i), weight*ParentB.S.item((i,i)) + (1-weight)*ParentA.S.item((i,i)))
		else:
			#kopiowanie rodzic�w, je�li crossover nie zaszed�
			ChildA = copy.deepcopy(ParentA);
			ChildB = copy.deepcopy(ParentB);
		return ChildA, ChildB
	
	@staticmethod
	def calculate_fitness(M, specimen):
		A = np.dot(np.dot(specimen.U, specimen.S), specimen.Vt)
		Res = np.subtract(M, A)
		norm = np.linalg.norm(Res)
		if norm == 0:
			return sys.float_info.max
		else:
			return abs(1/norm)

	@staticmethod
	def mutate_sigma(Sigma):
		rows = np.size(Sigma, 0)
		columns = np.size(Sigma, 1)
		range_of_diag = min(rows,columns)
		mutation_multiplier = np.random.rand()*1.99+0.01;
		position_of_mutation = np.random.randint(range_of_diag)
		Sigma_mutated = Sigma.copy()
		new_value = Sigma.item((position_of_mutation,position_of_mutation))*mutation_multiplier
		Sigma_mutated.itemset((position_of_mutation,position_of_mutation),new_value)
		return Sigma_mutated

	@staticmethod
	def mutate_orthogonal(ortho):
		rows = np.size(ortho, 0)
		mutation_matrix = ortho_group.rvs(rows)
		ortho_mutated = ortho.dot(mutation_matrix)
		return ortho_mutated

	@staticmethod
	def mutate(Specimen):
		if np.random.rand() < 0.01:
			Specimen.S = Genetic_extended.mutate_sigma(Specimen.S)
			Specimen.U = Genetic_extended.mutate_orthogonal(Specimen.U)
			Specimen.Vt = Genetic_extended.mutate_orthogonal(Specimen.Vt)
	
	@staticmethod
	def select_parents(fitness, sum_of_fitness):
		random1=np.random.rand()*sum_of_fitness
		random2=np.random.rand()*sum_of_fitness
		sum_of_fitness_to_compare1=0
		sum_of_fitness_to_compare2=0
		chosen_index1=0
		chosen_index2=0
		for i in range(len(fitness)):
			sum_of_fitness_to_compare1+=fitness[i]
			if(sum_of_fitness_to_compare1>random1):
				chosen_index1=i
				break
		for i in range(len(fitness)):
			sum_of_fitness_to_compare2+=fitness[i]
			if(sum_of_fitness_to_compare2>random2):
				chosen_index2=i
				break
		return chosen_index1, chosen_index2

	@staticmethod
	def calculate_svd(M):
		population = Genetic_extended.create_population(M)
		for i in range(1000):
			fitness = []
			best_fit = 0
			best_S = 0
			for j in range(len(population)):
				fitness.append(Genetic_extended.calculate_fitness(M, population[j]))
				if fitness[j] > best_fit:
					best_S = j
					best_fit = fitness[j]
			new_population = []
			sum_of_fitness = sum(fitness)
			for j in range(0, len(population), 2):
				indexA, indexB = Genetic_extended.select_parents(fitness, sum_of_fitness)
				ChildA, ChildB = Genetic_extended.crossover(population[indexA], population[indexB])
				new_population.append(ChildA)
				new_population.append(ChildB)
			for j in range(len(population)):
				Genetic_extended.mutate(new_population[j])
			rand_index = np.random.randint(len(new_population))
			new_population[rand_index] = copy.deepcopy(population[best_S])
			population = copy.deepcopy(new_population)
		fitness = []
		best_fit = 0
		best_S = 0
		for i in range(len(population)):
			fitness.append(Genetic_extended.calculate_fitness(M, population[i]))
			if fitness[i] > best_fit:
				best_fit = fitness[i]
				best_S = i
		return population[best_S], 1/best_fit