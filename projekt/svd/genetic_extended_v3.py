import numpy as np
import copy
from specimen import Specimen

#Algorytm genetyczny rozszerzony dla tworzenia wszystkich macierzy U, Sigma i V
class Genetic_extended_v3:
	@staticmethod
	def create_population(M, r):
		xmax=abs(max(M.min(), M.max(), key=abs))
		rows=np.size(M,0)
		columns=np.size(M,1)
		population = []
		for i in range(20):
			S=np.zeros((r,r))
			for j in range(r):
				S.itemset((j,j),np.random.rand()*xmax)
			S_diagonal = np.diagonal(S)
			S_diagonal = np.sort(S_diagonal)[::-1]
			for j in range(len(S_diagonal)):
				S.itemset((j,j),S_diagonal[j])
			U = (np.random.rand(rows, r)*2)-1
			Vt = (np.random.rand(r, columns)*2)-1
			item = Specimen(U, S, Vt)
			population.append(item)
		return population

	@staticmethod
	def crossover(ParentA, ParentB):
		if np.random.rand() < 0.8:
			weight = np.random.rand()
			rows = np.size(ParentA.S, 0)
			columns = np.size(ParentA.S, 1)
			#utworzenie obiekt�w dzieci
			Sa = weight*ParentA.S + (1-weight)*ParentB.S
			Sa_diagonal = np.diagonal(Sa)
			Sa_diagonal = np.sort(Sa_diagonal)[::-1]
			for j in range(len(Sa_diagonal)):
				Sa.itemset((j,j),Sa_diagonal[j])
			Sb = weight*ParentB.S + (1-weight)*ParentA.S
			Sb_diagonal = np.diagonal(Sb)
			Sb_diagonal = np.sort(Sb_diagonal)[::-1]
			for j in range(len(Sb_diagonal)):
				Sb.itemset((j,j),Sb_diagonal[j])
			Ua = weight*ParentA.U + (1-weight)*ParentB.U
			Ub = weight*ParentB.U + (1-weight)*ParentA.U
			Vta = weight*ParentA.Vt + (1-weight)*ParentB.Vt
			Vtb = weight*ParentB.Vt + (1-weight)*ParentA.Vt
			ChildA = Specimen(Ua, Sa, Vta)
			ChildB = Specimen(Ub, Sb, Vtb)
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
	def mutate_sigma(S):
		rows = np.size(S, 0)
		columns = np.size(S, 1)
		range_of_diag = min(rows,columns)
		mutation_multiplier = np.random.rand()*1.99+0.01;
		position_of_mutation = np.random.randint(range_of_diag)
		S_mutated = S.copy()
		new_value = S.item((position_of_mutation,position_of_mutation))*mutation_multiplier
		S_mutated.itemset((position_of_mutation,position_of_mutation),new_value)
		S_diagonal = np.diagonal(S_mutated)
		S_diagonal = np.sort(S_diagonal)[::-1]
		for j in range(len(S_diagonal)):
			S_mutated.itemset((j,j),S_diagonal[j])
		return S_mutated

	@staticmethod
	def mutate_orthogonal(ortho):
		rows = np.size(ortho, 0)
		columns = np.size(ortho, 1)
		mutation_multiplier = np.random.rand()*1.99+0.01;
		i = np.random.randint(rows)
		j = np.random.randint(columns)
		ortho_mutated = ortho.copy()
		new_value = ortho.item((i,j))*mutation_multiplier
		ortho_mutated.itemset((i,j),new_value)
		return ortho_mutated

	@staticmethod
	def mutate(Specimen):
		if np.random.rand() < 0.01:
			Specimen.S = Genetic_extended_v3.mutate_sigma(Specimen.S)
			Specimen.U = Genetic_extended_v3.mutate_orthogonal(Specimen.U)
			Specimen.Vt = Genetic_extended_v3.mutate_orthogonal(Specimen.Vt)
	
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
	def calculate_svd(M, r):
		population = Genetic_extended_v3.create_population(M, r)
		for i in range(100000):
			fitness = []
			best_fit = 0
			best_S = 0
			for j in range(len(population)):
				fitness.append(Genetic_extended_v3.calculate_fitness(M, population[j]))
				if fitness[j] > best_fit:
					best_S = j
					best_fit = fitness[j]
			new_population = []
			sum_of_fitness = sum(fitness)
			for j in range(0, len(population), 2):
				indexA, indexB = Genetic_extended_v3.select_parents(fitness, sum_of_fitness)
				ChildA, ChildB = Genetic_extended_v3.crossover(population[indexA], population[indexB])
				new_population.append(ChildA)
				new_population.append(ChildB)
			for j in range(len(population)):
				Genetic_extended_v3.mutate(new_population[j])
			rand_index = np.random.randint(len(new_population))
			new_population[rand_index] = copy.deepcopy(population[best_S])
			population = copy.deepcopy(new_population)
		fitness = []
		best_fit = 0
		best_S = 0
		for i in range(len(population)):
			fitness.append(Genetic_extended_v3.calculate_fitness(M, population[i]))
			if fitness[i] > best_fit:
				best_fit = fitness[i]
				best_S = i
		return population[best_S], 1/best_fit