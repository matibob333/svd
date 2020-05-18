import numpy as np
import copy

class Genetic:
	@staticmethod
	def create_population(M):
		xmax=abs(max(M.min(), M.max(), key=abs))
		rows=np.size(M,0)
		columns=np.size(M,1)
		population = []
		for i in range(20):
			temp=np.zeros((rows,columns))
			for j in range(min(rows,columns)):
				temp.itemset((j,j),np.random.rand()*xmax)
			population.append(temp)
		return population

	@staticmethod
	def crossover(ParentA, ParentB):
		if np.random.rand() < 0.8:
			weight = np.random.rand()
			rows = np.size(ParentA, 0)
			columns = np.size(ParentA, 1)
			ChildA = np.zeros((rows, columns))
			ChildB = np.zeros((rows, columns))
			#for i in range(min(rows, columns)):
			#	ChildA.itemset((i,i), weight*ParentA.item((i,i)) + (1-weight)*ParentB.item((i,i)))
			#	ChildB.itemset((i,i), weight*ParentB.item((i,i)) + (1-weight)*ParentA.item((i,i)))
			ChildA = weight*ParentA + (1-weight)*ParentB
			ChildB = weight*ParentB + (1-weight)*ParentA
		else:
			ChildA = ParentA.copy();
			ChildB = ParentB.copy();
		return ChildA, ChildB

	@staticmethod
	def calculate_fitness(M, U, S, Vt):
		A = np.dot(np.dot(U, S), Vt)
		Res = np.subtract(M, A)
		norm = np.linalg.norm(Res)
		if norm == 0:
			return sys.float_info.max
		else:
			return abs(1/norm)

	@staticmethod
	def mutate(Child):
		if np.random.rand() < 0.01:
			rows = np.size(Child, 0)
			columns = np.size(Child, 1)
			range_of_diag = min(rows,columns)
			mutation_multiplier = np.random.rand()*1.99+0.01;
			position_of_mutation = np.random.randint(range_of_diag)
			Child_mutated = Child.copy()
			new_value = Child.item((position_of_mutation,position_of_mutation))*mutation_multiplier
			Child_mutated.itemset((position_of_mutation,position_of_mutation),new_value)
		else:
			Child_mutated = Child.copy()
		return Child_mutated
	
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
	def calculate_sigma_with_UVt(M, U, Vt):
		population = Genetic.create_population(M)
		for i in range(1000):
			fitness = []
			best_fit = 0
			best_S = 0
			for j in range(len(population)):
				fitness.append(Genetic.calculate_fitness(M, U, population[j], Vt))
				if fitness[j] > best_fit:
					best_S = j
					best_fit = fitness[j]
			new_population = []
			sum_of_fitness = sum(fitness)
			for j in range(0, len(population), 2):
				indexA, indexB = Genetic.select_parents(fitness, sum_of_fitness)
				ChildA, ChildB = Genetic.crossover(population[indexA], population[indexB])
				new_population.append(ChildA)
				new_population.append(ChildB)
			for j in range(len(population)):
				new_population[j] = Genetic.mutate(new_population[j])
			rand_index = np.random.randint(len(new_population))
			new_population[rand_index] = population[best_S].copy()
			population = copy.deepcopy(new_population)
		fitness = []
		best_fit = 0
		best_S = 0
		for i in range(len(population)):
			fitness.append(Genetic.calculate_fitness(M, U, population[i], Vt))
			if fitness[i] > best_fit:
				best_fit = fitness[i]
				best_S = i
		return population[best_S], 1/best_fit