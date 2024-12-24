# Problem_1
Best Expression Found: 
sin(x0 min x0)
Final Fitness: 0.996016
Execution Time: 47.06 seconds
Generations: 300
Performance Metrics:
Mean Squared Error: 0.000000
R² Score: 1.000000 (100.0% of variance explained)

# Problem_2
params = GeneticParams(
            tournament_size=8,
            mutation_prob=0.8,
            crossover_prob=0.7,
            elitism_count=10,
            population_size=1000,
            generations=500,
            maximum_tree_depth=7,
            minimum_tree_depth=2,
            depth_penalty_threshold=5,  # Depth at which penalties start
            max_tree_size=15,
            size_penalty_threshold=5,  # Size at which penalties start
            parsimony_coefficient=0.1,  # Controls size penalty weight
            unused_var_coefficient=0.1,  # Coefficient for unused variable penalty
        )

==================================================
==== SYMBOLIC REGRESSION RESULTS - problem_2 =====
==================================================
Best Expression Found: sqrt(((4.691 // (3.457 ** (-4.374 - x0))) // ((sign(abs(x1)) atan2 x2) // max(x1, x0))))
Final Fitness: 0.195049
Execution Time: 406.00 seconds
Generations: 500
==================================================
Performance Metrics:
Mean Squared Error: 20783913855133.996094
R² Score: 0.298168 (29.8% of variance explained)
==================================================

# Problem_3
Best Expression Found: 
cosh(x0) - x1 abs_diff abs(sqrt(arcsin(sqrt(log(sigmoid(x2) - x1 % x1))) - x2)) min sqrt(cosh(x1)) * (x1 + x1) + x1 * sinh(x1) - sin(x1) // x1
Final Fitness: 0.010168
Execution Time: 280.18 seconds
Generations: 300
Performance Metrics:
Mean Squared Error: 97.311628
R² Score: 0.962010 (96.2% of variance explained)

# Problem_4
if params is None:
        params = GeneticParams(
            tournament_size=7,
            mutation_prob=0.6,
            crossover_prob=0.8,
            elitism_count=8,
            population_size=1200,
            generations=500,
            maximum_tree_depth=7,
            minimum_tree_depth=2,
            depth_penalty_threshold=5,  # Depth at which penalties start
            max_tree_size=15,
            size_penalty_threshold=5,  # Size at which penalties start
            parsimony_coefficient=0.1,  # Controls size penalty weight
            unused_var_coefficient=0.5,  # Coefficient for unused variable penalty
        )

==================================================
==== SYMBOLIC REGRESSION RESULTS - problem_4 =====
==================================================
Best Expression Found: (sinh((log(3.237) * (cos(x0) - (x0 // x0)))) + exp(|cos(x0) - reciprocal(-0.727)|))
Final Fitness: 0.831752
Execution Time: 471.78 seconds
Generations: 500
==================================================
Performance Metrics:
Mean Squared Error: 0.307729
R² Score: 0.985768 (98.6% of variance explained)
==================================================

# Problem_5
Best Expression Found: 
x0 // x0 + x1
Final Fitness: 0.995025
Execution Time: 71.58 seconds
Generations: 300
Performance Metrics:
Mean Squared Error: 0.000000
R² Score: -0.064122 (-6.4% of variance explained)

# Problem_6
if params is None:
        params = GeneticParams(
            tournament_size=7,
            mutation_prob=0.4,
            crossover_prob=0.7,
            elitism_count=20,
            population_size=1200,
            generations=500,
            maximum_tree_depth=5,
            minimum_tree_depth=2,
            depth_penalty_threshold=5,  # Depth at which penalties start
            max_tree_size=15,
            size_penalty_threshold=5,  # Size at which penalties start
            parsimony_coefficient=0.1,  # Controls size penalty weight
            unused_var_coefficient=0.5,  # Coefficient for unused variable penalty
        )

==================================================
==== SYMBOLIC REGRESSION RESULTS - problem_6 =====
==================================================
Best Expression Found: (x1 + (sqrt(cot(log2(2.177))) * (sqrt(-4.419) + (x1 - x0))))
Final Fitness: 0.95037
Execution Time: 165.57 seconds
Generations: 500
==================================================
Performance Metrics:
Mean Squared Error: 0.000020
R² Score: 0.999999 (100.0% of variance explained)
==================================================

# Problem_7
Best Expression Found: 
exp(arcsin(cos(x1 abs_diff x0)) * sqrt(arcsin(exp(x1) * x1 // x1) * (cos(x0) max x0 + x1 + x0 + x1)) max x1 * sqrt(x0) % arcsin(x1) max x0 max arcsin(reciprocal(sqrt(x0))) - x0 max x1)
Final Fitness: 0.0100248
Execution Time: 239.96 seconds
Generations: 300
Performance Metrics:
Mean Squared Error: 98.707621
R² Score: 0.861201 (86.1% of variance explained)

# Problem_8
params = GeneticParams(
            tournament_size=7,
            mutation_prob=0.8,
            crossover_prob=0.7,
            elitism_count=8,
            population_size=1000,
            generations=300,
            maximum_tree_depth=7,
            minimum_tree_depth=2,
            depth_penalty_threshold=5,  # Depth at which penalties start
            max_tree_size=15,
            size_penalty_threshold=5,  # Size at which penalties start
            parsimony_coefficient=0.1,  # Controls size penalty weight
            unused_var_coefficient=0.9,  # Coefficient for unused variable penalty
        )

==================================================
==== SYMBOLIC REGRESSION RESULTS - problem_8 =====
==================================================
Best Expression Found: ((sinh(4.067) // exp(x5)) * -3.664)
Final Fitness: 0.34403
Execution Time: 707.74 seconds
Generations: 300
==================================================
Performance Metrics:
Mean Squared Error: 10252071.589658
R² Score: 0.548485 (54.8% of variance explained)
==================================================