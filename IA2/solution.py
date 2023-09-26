import random
import math
import time
import matplotlib.pyplot as plt
import sys
import numpy as np

#   We use seed to generate same test each time running
random.seed(40)

#   support function

class City:
    def __init__(self, x, y) -> None:
        self.x = x  # Assign x-coordinate
        self.y = y  # Assign y-coordinate
    
    def distance(self, city):
        #   Calculate distance between two cites
        return ((self.x - city.x) ** 2 + (self.y - city.y) ** 2) ** 0.5
    
def get_tour_length(tour):
    #   Calculate length of tour
    return sum(tour[i].distance(tour[i - 1]) for i in range(len(tour)))
def create_random_cities(n, x_range=(0, 100), y_range=(0, 100)):
    # Create test case
    city_coordinates = set()
    
    while len(city_coordinates) < n:
        x = random.randint(*x_range)
        y = random.randint(*y_range)
        city_coordinates.add((x, y))
    
    return [City(x, y) for x, y in city_coordinates]

#   Design graph
def plot_tour(tour, title):
    x_coords = [city.x for city in tour]
    y_coords = [city.y for city in tour]

    plt.figure()
    plt.scatter(x_coords, y_coords, marker='o', color='blue')
    plt.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], linestyle='-', color='red')
    
    for i, city in enumerate(tour):
        plt.text(city.x + 1, city.y + 1, f"{i}", fontsize=10)

    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title(title)
    plt.grid()
    plt.show()

#   main algorithm

#   Genetic Algorithm
#   Initial Parameter:
#   - cities: The list of cities
#   - population_size: The number of chromosomes in the population
#   - nb_gens: The number of generations before algorithm stop
#   - mutation_rate: The probability of mutation for swap mutation


class GA:
    def __init__(self, cities, population_size, nb_gens, mutation_rate) -> None:
        self.cities = cities
        self.population_size = population_size
        self.nb_gens = nb_gens
        self.mutation_rate = mutation_rate
    
    def Initialization(self, cities, size):
        #   Initialize population by using set of city permutation
        return [random.sample(cities, len(cities)) for _ in range(size)]
    
    #   Roulette_Wheel_Selection
    def Roulette_Wheel_Selection(self, population):
        #   Calculate the fitness for each chromosome in the population
        fitnesses = [1 / get_tour_length(chromosome) for chromosome in population]
        total = sum(fitnesses)

        # Calculate the selection probabilities based on fitnesses
        probabilities = [f / total for f in fitnesses]

        # Select two parents using roulette wheel selection
        return random.choices(population, weights=probabilities, k=2) 
    
    # Rank Selction
    def Rank_Selection(self, population):
        #   Calculate the fitness for each chromosome in the population
        fitnesses = [1 / get_tour_length(chromosome) for chromosome in population]

        # Rank the chromosomes based on their fitness
        ranked_chromosomes = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)

        # Calculate the selection probabilities based on the ranks
        total_chromosomes = len(population)
        probabilities = [(total_chromosomes - rank) / ((total_chromosomes * (total_chromosomes + 1)) / 2) for rank in range(total_chromosomes)]

         # Select two parents using rank selection
        return random.choices([chromosome for chromosome, _ in ranked_chromosomes], weights=probabilities, k=2)

    def Crossover(self, parent1, parent2):
        #   Two point crossover
        #   Choose two random points
        start, end = sorted(random.sample(range(len(parent1)), 2))

        #   Generate children by selected parents's segment
        child1 = parent1[start:end + 1] + [parent for parent in parent2 if parent not in parent1[start:end + 1]]
        child2 = parent2[start:end + 1] + [parent for parent in parent1 if parent not in parent2[start:end + 1]]

        return child1, child2
    
    def Mutation(self, chromosome, rate):
        #   Swap mutation
        #   we Choose two Point - City and we switch them
        for i in range(len(chromosome)):
            if random.random() < rate:
                j = random.randint(0, len(chromosome) - 1)
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

        return chromosome
    
    def Mutation_Scramble(self, chromosome):
        #   we choose a random segment in The Current Chromosome and we interchange the values
        #   Choose two random mutation points
        start, end = sorted(random.sample(range(len(chromosome)), 2))

        #   Shuffle the selected subsequence
        subsequence = chromosome[start:end + 1]
        random.shuffle(subsequence)
        chromosome[start:end + 1] = subsequence

        return chromosome

    def execute(self):
        #   Initialize population
        population = self.Initialization(self.cities, self.population_size)

        #   Execute algorithm in limited generations
        for _ in range(self.nb_gens):
            children = []

            #   Generate childs until reach desired size
            while len(children) < len(population):
                #   Select parent using roulette wheel selection
                #parent1, parent2 = self.Roulette_Wheel_Selection(population)

                #   Select parent using rank selection   
                parent1, parent2 = self.Rank_Selection(population)

                #   Product children
                child1, child2 = self.Crossover(parent1, parent2)

                #   Mutation
                #   Swap Mutation
                #children.append(self.Mutation(child1, self.mutation_rate))
                #children.append(self.Mutation(child2, self.mutation_rate))

                #   Scramble Mutation
                children.append(self.Mutation_Scramble(child1))
                children.append(self.Mutation_Scramble(child2))
            
            #   Replace the old population with the new one
            population = children
        #   Find best way 
        best_tour = min(population, key=get_tour_length)

        #   Best length
        best_length = get_tour_length(best_tour)

        #   Return best result
        return best_tour, best_length


#   Ant Colony Algorithm
#   Initial Parameter
#   -   cities: The list of cities
#   -   nb_ants: The number of ants in the colony
#   -	nb_iters: number of loop algorithm will work
#   -	alpha: The pheromone influence
#   -	beta: The heuristic information influence
#   -	rho: The pheromone evaporation rate
#   -	Q: The correction parameter
class ACO:
    def __init__(self, cities, nb_ants, nb_iters, alpha, beta, rho, Q) -> None:
        self.cities = cities
        self.nb_ants = nb_ants
        self.nb_iters = nb_iters
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.nb_cities = len(cities)
        pass

    def Calculate_Prob(self, cur_city, next_city, tour):
        #   Calculate the probabilities for moving from the current city to next considering city    
        if self.cities[next_city] in tour:
            return 0
        #   Calculate probability base on pheromone and distance
        prob = 0
        try:
            prob = (self.matrix[cur_city, next_city] ** self.alpha) * ((1 / self.cities[cur_city].distance(self.cities[next_city])) ** self.beta)
        except:
            prob = 0
        return prob
    

    def Find_Next(self, cur_city, tour):
        #   Calculate the probabilities for moving from the current city to each other 
        moving_prob = []
        for city in range(self.nb_cities):
            moving_prob.append(self.Calculate_Prob(cur_city, city, tour))
        probs = []
        t = sum(moving_prob)
        #   Normalize the probabilities
        for p in moving_prob:
            probs.append(p / t)
        # Select the next city using the probabilities
        return self.cities[np.random.choice(self.nb_cities, p=probs)]

    def Setup_Tour(self):
        tour = []
        #   Start with random point
        tour.append(random.choice(self.cities)) 
        #   Find next city base on probability

        for i in range(self.nb_cities - 1):
            tour.append(self.Find_Next(self.cities.index(tour[-1]), tour))

        return tour

    def Update(self, ant_tours, tour_lengths):
        #   Calculate pheromone changing levels based on the tours and their lengths
        delta = np.zeros((self.nb_cities, self.nb_cities))

        #   After completing the city tour. An ant that sends an amount of pheramon denta on each edge of its route
        for tour, length in zip(ant_tours, tour_lengths):
            for i in range(self.nb_cities):
                delta[self.cities.index(tour[i - 1]), self.cities.index(tour[i])] += self.Q / length

        return (1 - self.rho) * self.matrix + delta 


    def execute(self):
        #   Initialize pheromone matrix
        self.matrix = np.ones((self.nb_cities, self.nb_cities))

        best_tour = None  
        best_length = float('inf')  

        for _ in range(self.nb_iters):
            #   Initialize tours for all of ants
            ant_tours = []
            for i in range(self.nb_ants):
                ant_tours.append(self.Setup_Tour())

            #   Calculate length of each tour
            tour_lengths = []
            for tour in ant_tours:
                tour_lengths.append(get_tour_length(tour))
            
            #   Update pheromone matrix
            self.matrix = self.Update(ant_tours, tour_lengths)

            #   Check best tour
            cur_best_tour = min(ant_tours, key=get_tour_length)
            cur_best_len = get_tour_length(cur_best_tour)

            if cur_best_len < best_length:
                best_tour = cur_best_tour
                best_length = cur_best_len
        
        return best_tour, best_length


#   MAIN
def GA_test():
    list_cites = [(63, 8), (33, 34), (27, 21), (78, 19), (30, 77), (9, 34), (23, 8), (55, 7), (73, 56), (10, 71)]
    cities = [City(i[0], i[1]) for i in list_cites]

    start_time = time.time()
    ga = GA(cities, population_size=100, nb_gens=500, mutation_rate=0.1)
    best_tour, best_length = ga.execute()

    print("Best length:\t\t" + str(best_length))
    final_tour = [(city.x, city.y) for city in best_tour]
    print("Best tour:\t\t")
    print(final_tour)

    plot_tour(best_tour, "GA Tour with Time: " + str(time.time() - start_time))

def ACO_test():
    list_cites = [(63, 8), (33, 34), (27, 21), (78, 19), (30, 77), (9, 34), (23, 8), (55, 7), (73, 56), (10, 71)]
    cities = [City(i[0], i[1]) for i in list_cites]

    start_time = time.time()
    aco = ACO(cities, nb_ants=30, nb_iters=200, alpha=1, beta=3, rho=0.1, Q=200)
    best_tour, best_length = aco.execute()

    print("Best length:\t\t" + str(best_length))
    final_tour = [(city.x, city.y) for city in best_tour]
    print("Best tour:\t\t")
    print(final_tour)

    plot_tour(best_tour, "ACO Tour with Time: " + str(time.time() - start_time))

def main_test():
    graph_sizes = list(range(10, 51, 5))
    total_aco_times = []
    aco_tour_lengths = []
    total_ga_times = []
    ga_tour_lengths = []

    for n in graph_sizes:
        #   Create test case
        cities = create_random_cities(n)
        # x = []
        # for city in cities:
        #     x.append((city.x, city.y))
        # print(x)

        start_time = time.time()
        aco = ACO(cities, nb_ants=30, nb_iters=200, alpha=1, beta=3, rho=0.1, Q=200)
        best_tour, best_length = aco.execute()
        total_aco_times.append(time.time() - start_time)
        aco_tour_lengths.append(best_length)

        start_time = time.time()
        ga = GA(cities, population_size=100, nb_gens=500, mutation_rate=0.1)
        best_tour, best_length = ga.execute()
        total_ga_times.append(time.time() - start_time)
        ga_tour_lengths.append(best_length)


    plt.figure()
    plt.plot(graph_sizes, total_aco_times, label="ACO")
    plt.plot(graph_sizes, total_ga_times, label="GA")
    plt.xlabel("Number of cities")
    plt.ylabel("Execution time (seconds)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(graph_sizes, aco_tour_lengths, label="ACO")
    plt.plot(graph_sizes, ga_tour_lengths, label="GA")
    plt.xlabel("Number of cities")
    plt.ylabel("Tour length")
    plt.legend()
    plt.show()

def main() -> int:
    #   test case
    #GA_test()
    #ACO_test()
    main_test()
    
    

    return 0

if __name__ == '__main__':
    sys.exit(main()) 