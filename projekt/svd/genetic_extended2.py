import numpy as np
import copy


class Genetic_extended2:
	@staticmethod
	def create_population(M):
		population = []
		for i in range(50):
			item = np.random.rand(np.size(M,1),1)*2-1
			item = np.divide(item,np.linalg.norm(item))
			population.append(item)
		return population

	@staticmethod
	def calculate_fitness(M,vec):
		return np.linalg.norm(np.dot(M,vec))

	@staticmethod
	def calculate_last_parameters(V, number_of_vars, vec):
		A = (V[0:number_of_vars,:]).T
		x = vec[0:number_of_vars,:]
		b = -(A.dot(x))
		B = (V[number_of_vars:,:]).T
		y = np.linalg.solve(B,b)
		ret = np.copy(vec)
		for i in range(number_of_vars,np.size(ret,0)):
			ret.itemset((i,0),y.item((i-number_of_vars,0)))
		return ret

	@staticmethod
	def crossover(ParentA, ParentB):
		if np.random.rand() < 0.8:
			weight = np.random.rand()
			ChildA = np.add(np.multiply(ParentA,weight), np.multiply(ParentB, (1-weight)))
			ChildB = np.add(np.multiply(ParentB,weight), np.multiply(ParentA, (1-weight)))
			ChildA = np.divide(ChildA,np.linalg.norm(ChildA))
			ChildB = np.divide(ChildB,np.linalg.norm(ChildB))
		else:
			ChildA = copy.deepcopy(ParentA)
			ChildB = copy.deepcopy(ParentB)
		return ChildA, ChildB

	@staticmethod
	def mutate(vec, number_of_vars):
		mvec = copy.deepcopy(vec)
		if np.random.rand() < 0.01:
			mutation_multiplier = np.random.rand()*1.99+0.01;
			position_of_mutation = np.random.randint(number_of_vars)
			mvec.itemset((position_of_mutation,0),vec.item((position_of_mutation,0))*mutation_multiplier);
			mvec = np.divide(vec,np.linalg.norm(mvec))
		return mvec

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
	def find_next_vector(M, V, number_of_vars):
		population = Genetic_extended2.create_population(M);
		for i in range(len(population)):
			population[i] = Genetic_extended2.calculate_last_parameters(V, number_of_vars,population[i])
			norm = np.linalg.norm(population[i])
			population[i] = np.divide(population[i],norm)
		for i in range(1000):
			fitness = []
			last_best = -1
			last_best_fit = -1
			for j in range(len(population)):
				fitness.append(Genetic_extended2.calculate_fitness(M,population[j]))
				if fitness[j] > last_best_fit:
					last_best_fit = fitness[j]
					last_best = j
			new_population = []
			sum_of_fitness = sum(fitness)
			for j in range(0, len(population), 2):
				indexA, indexB = Genetic_extended2.select_parents(fitness, sum_of_fitness)
				ChildA, ChildB = Genetic_extended2.crossover(population[indexA], population[indexB])
				new_population.append(ChildA)
				new_population.append(ChildB)
			for j in range(len(population)):
				norm = np.linalg.norm(new_population[j])
				new_population[j] = np.divide(new_population[j],norm)
				new_population[j] = Genetic_extended2.mutate(new_population[j], number_of_vars)
				new_population[j] = Genetic_extended2.calculate_last_parameters(V, number_of_vars, new_population[j])
				norm = np.linalg.norm(new_population[j])
				new_population[j] = np.divide(new_population[j],norm)
			rand_index = np.random.randint(len(new_population))
			new_population[rand_index] = copy.deepcopy(population[last_best])
			population = copy.deepcopy(new_population)
		best_fit = -1
		best_index = -1
		for i in range(len(population)):
			fit = Genetic_extended2.calculate_fitness(M,population[i])
			if fit > best_fit:
				best_fit = fit
				best_index = i
		return np.divide(population[best_index],np.linalg.norm(population[best_index]))

	@staticmethod
	def find_highest_sigma_vector(M):
		population = Genetic_extended2.create_population(M);
		for i in range(len(population)):
			norm = np.linalg.norm(population[i])
			population[i] = np.divide(population[i],norm)
		for i in range(1000):
			fitness = []
			last_best = -1
			last_best_fit = -1
			for j in range(len(population)):
				fitness.append(Genetic_extended2.calculate_fitness(M,population[j]))
				if fitness[j] > last_best_fit:
					last_best_fit = fitness[j]
					last_best = j
			new_population = []
			sum_of_fitness = sum(fitness)
			for j in range(0, len(population), 2):
				indexA, indexB = Genetic_extended2.select_parents(fitness, sum_of_fitness)
				ChildA, ChildB = Genetic_extended2.crossover(population[indexA], population[indexB])
				new_population.append(ChildA)
				new_population.append(ChildB)
			for j in range(len(population)):
				norm = np.linalg.norm(new_population[j])
				new_population[j] = np.divide(new_population[j],norm)
				new_population[j] = Genetic_extended2.mutate(new_population[j], np.size(new_population[j],0))
				norm = np.linalg.norm(new_population[j])
				new_population[j] = np.divide(new_population[j],norm)
			rand_index = np.random.randint(len(new_population))
			new_population[rand_index] = copy.deepcopy(population[last_best])
			population = copy.deepcopy(new_population)
		best_fit = -1
		best_index = -1
		for i in range(len(population)):
			fit = Genetic_extended2.calculate_fitness(M,population[i])
			if fit > best_fit:
				best_fit = fit
				best_index = i
		return np.divide(population[best_index],np.linalg.norm(population[best_index]))

	@staticmethod
	def calculate_svd(M, param=-1):
		if param == -1:
			param = min(np.size(M, 0), np.size(M, 1))
		C = np.copy(M)
		for i in range(np.size(C,0)):
			for j in range(np.size(C,1)):
				if(C.item((i,j))==0):
					C.itemset((i,j),1e-6)
		V = Genetic_extended2.find_highest_sigma_vector(C)
		U = C.dot(V)
		S = []
		s = np.linalg.norm(U)
		U = np.divide(U,s)
		S.append(s)
		for i in range(1, param):
			v = Genetic_extended2.find_next_vector(C,V,np.size(V,0)-i)
			u = C.dot(v)
			s = np.linalg.norm(u)
			u = np.divide(u,s)
			V = np.c_[V, v]
			U = np.c_[U, u]
			S.append(s)
		Sdiag = np.zeros((len(S),len(S)))
		for i in range(len(S)):
			Sdiag.itemset((i,i),S[i])
		return U, Sdiag, V