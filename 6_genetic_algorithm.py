# """
# Program 6: Genetic Algorithm (Maximize a function)
# Time Complexity: O(g * p * n) where g = generations, p = population, n = chromosome length
# Space Complexity: O(p * n)
# """

# import random

# def fitness(chromosome):
#     """Fitness function: maximize sum of bits (OneMax problem)."""
#     return sum(chromosome)

# def create_population(pop_size, chrom_length):
#     """Generate random initial population."""
#     return [[random.randint(0, 1) for _ in range(chrom_length)] for _ in range(pop_size)]

# def tournament_selection(population, fitnesses, k=3):
#     """Select parent using tournament selection."""
#     indices = random.sample(range(len(population)), k)
#     best = max(indices, key=lambda i: fitnesses[i])
#     return population[best][:]

# def crossover(parent1, parent2, rate=0.8):
#     """Single-point crossover."""
#     if random.random() < rate:
#         point = random.randint(1, len(parent1) - 1)
#         child1 = parent1[:point] + parent2[point:]
#         child2 = parent2[:point] + parent1[point:]
#         return child1, child2
#     return parent1[:], parent2[:]

# def mutate(chromosome, rate=0.01):
#     """Bit-flip mutation."""
#     for i in range(len(chromosome)):
#         if random.random() < rate:
#             chromosome[i] = 1 - chromosome[i]
#     return chromosome

# def genetic_algorithm(pop_size=50, chrom_length=20, generations=100,
#                       crossover_rate=0.8, mutation_rate=0.02):
#     """Run the genetic algorithm."""
#     population = create_population(pop_size, chrom_length)

#     for gen in range(generations):
#         fitnesses = [fitness(ind) for ind in population]
#         best_fitness = max(fitnesses)
#         best_individual = population[fitnesses.index(best_fitness)]

#         # Check if optimal solution found
#         if best_fitness == chrom_length:
#             print(f"  Optimal solution found at generation {gen}")
#             return best_individual, best_fitness

#         # Create new population with elitism (keep best)
#         new_population = [best_individual[:]]

#         while len(new_population) < pop_size:
#             parent1 = tournament_selection(population, fitnesses)
#             parent2 = tournament_selection(population, fitnesses)
#             child1, child2 = crossover(parent1, parent2, crossover_rate)
#             new_population.append(mutate(child1, mutation_rate))
#             if len(new_population) < pop_size:
#                 new_population.append(mutate(child2, mutation_rate))

#         population = new_population

#         if gen % 20 == 0:
#             avg_fitness = sum(fitnesses) / len(fitnesses)
#             print(f"  Gen {gen:3d}: Best = {best_fitness}, Avg = {avg_fitness:.2f}")

#     fitnesses = [fitness(ind) for ind in population]
#     best_idx = fitnesses.index(max(fitnesses))
#     return population[best_idx], fitnesses[best_idx]


# # Driver Code
# if __name__ == "__main__":
#     print("=== Genetic Algorithm: OneMax Problem ===")
#     print("Goal: Maximize number of 1s in a binary string of length 20\n")

#     random.seed(42)
#     best, best_fit = genetic_algorithm(
#         pop_size=50,
#         chrom_length=20,
#         generations=100,
#         crossover_rate=0.8,
#         mutation_rate=0.02
#     )
#     print(f"\nBest Solution: {''.join(map(str, best))}")
#     print(f"Fitness: {best_fit}/20")


# 