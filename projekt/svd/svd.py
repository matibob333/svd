import numpy as np
from genetic import Genetic
from genetic_extended import Genetic_extended
from genetic_extended_v2 import Genetic_extended_v2
from genetic_extended_v3 import Genetic_extended_v3
from genetic_extended_v4 import Genetic_extended_v4
from gradient import Gradient
from gradient_v2 import Gradient_v2
from gradient_v3 import Gradient_v3
from specimen import Specimen

M = np.zeros((4,5))
M.itemset((0,0), 1)
M.itemset((0,4), 2)
M.itemset((1,2), 3)
M.itemset((3,1), 4)

U = np.zeros((4,4))
U.itemset((0,2), 1)
U.itemset((1,1), 1)
U.itemset((2,3), -1)
U.itemset((3,0), 1)



Vt = np.zeros((5,5))
Vt.itemset((0,1),1)
Vt.itemset((1,2),1)
Vt.itemset((2,0),0.2**0.5)
Vt.itemset((2,4),0.8**0.5)
Vt.itemset((3,3),1)
Vt.itemset((4,0),0.8**0.5*(-1))
Vt.itemset((4,4),0.2**0.5)

#S, norm = Genetic.calculate_sigma_with_UVt(M, U, Vt)
#print(S)
#print(norm)


#specimen, norm = Genetic_extended_v4.calculate_svd(M, 4)
#product = specimen.calculate_product()

B = np.zeros((2,2))
B.itemset((0,0),1)
B.itemset((0,1),4)
B.itemset((1,0),-3)
B.itemset((1,1),-2)

A = np.matrix('5,4,2;3,2,2;5,1,2')


U1, S1, V1 = Gradient_v3.calculate_svd(A)
product = U1.dot(S1).dot(V1.T)
a = 1