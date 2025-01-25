#include "GeneticAlgorithm.h"
#include "KMeans.h"

#include <iostream>
using namespace std;

GeneticAlgorithm::GeneticAlgorithm(int populationSize, const vector<CPoint>& points, int maxClusters,
    double mutationProbability, double crossoverProbability, int maxIterations, mt19937& randomEngine) 
    : populationSize(populationSize), points(points), maxClusters(maxClusters),
    mutationProbability(mutationProbability), crossoverProbability(crossoverProbability),
    maxIterations(maxIterations), randomEngine(randomEngine), bestIndividual() {
    numberOfPoints = points.size();
    population.resize(populationSize);
    fitness.resize(populationSize);
    initializePointDistances();
}

void GeneticAlgorithm::initializePointDistances() {
    pointDistances.resize(points.size() - 1);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < points.size() - 1; ++i) {
        pointDistances[i].resize(points.size() - 1 - i);
        for (size_t pointIndex = i + 1; pointIndex < points.size(); ++pointIndex) {
            pointDistances[i][pointIndex - 1 - i] = points[i].dCalculateDistance(points[pointIndex]);
        }

        //cout << pointDistances[i][0] << endl;
    }
}







void GeneticAlgorithm::initializePopulation() {
    //population.clear();
    //for (int i = 0; i < populationSize; ++i) {
    //    population.emplace_back(move(Individual(maxClusters, randomEngine, points)));
    //}

    int islandPopulationSize = populationSize / 8;

    islands.resize(8);
    islands[0] = new Island(IslandType::EXPLOITATIVE, islandPopulationSize, 0.01, 0.39, &Island::tournamentSelection,
        randomEngine, pointDistances, points, maxClusters);
    islands[1] = new Island(IslandType::EXPLOITATIVE, islandPopulationSize, 0.02, 0.28, &Island::tournamentSelection,
        randomEngine, pointDistances, points, maxClusters);
    islands[2] = new Island(IslandType::EXPLOITATIVE, islandPopulationSize, 0.03, 0.33, &Island::tournamentSelection,
        randomEngine, pointDistances, points, maxClusters);
    islands[3] = new Island(IslandType::EXPLORATIVE, islandPopulationSize, 0.3, 0.55, &Island::rouletteSelectionWorst,
        randomEngine, pointDistances, points, maxClusters);
    islands[4] = new Island(IslandType::EXPLORATIVE, islandPopulationSize, 0.34, 0.67, &Island::rouletteSelectionWorst,
        randomEngine, pointDistances, points, maxClusters);
    islands[5] = new Island(IslandType::EXPLORATIVE, islandPopulationSize, 0.32, 0.59, &Island::rouletteSelectionWorst,
        randomEngine, pointDistances, points, maxClusters);
    islands[6] = new Island(IslandType::EXPLORATIVE, islandPopulationSize, 0.28, 0.63, &Island::rouletteSelectionWorst,
        randomEngine, pointDistances, points, maxClusters);
    islands[7] = new Island(IslandType::EXPLORATIVE, islandPopulationSize, 0.36, 0.68, &Island::rouletteSelectionWorst,
        randomEngine, pointDistances, points, maxClusters);

    exploitativeIslands.resize(3);
    exploitativeIslands[0] = islands[0];
    exploitativeIslands[1] = islands[1];
    exploitativeIslands[2] = islands[2];

    explorativeIslands.resize(5);
    explorativeIslands[0] = islands[3];
    explorativeIslands[1] = islands[4];
    explorativeIslands[2] = islands[5];
    explorativeIslands[3] = islands[6];
    explorativeIslands[4] = islands[7];
}


void GeneticAlgorithm::run() {

    int migrationFromExploitative = 45;         // 150
    int migrationFromExplorative = 15;   // 50
    int stagnatingIterations = 30;      // 100
    for (size_t iteration = 0; iteration < maxIterations; ++iteration) {
        #pragma omp parallel for
        for (size_t i = 0; i < islands.size(); ++i) {
            islands[i]->runGeneration(iteration, maxIterations);
        }

        if (iteration % migrationFromExploitative == 0) {
            performMigrationExploitative(0.12);
        }

        if (iteration % migrationFromExplorative == 0) {
            performMigrationExplorative(0.20);
        }

        // 1. Zwiêksz czêstotliwoœæ migracji z eksplorac, jeœli wykryta zostanie stagnacja w wyspie eksploatacyjnej
        // 2. po migracji zidentyfikuj nowo przyby³e osobniki i wykonaj na nich intensywn¹ lokaln¹ optymalizacjê.
        // 3. wiecej losowych
        // 4. wieksza mutacja 

        #pragma omp parallel for
        for (size_t i = 0; i < islands.size(); ++i) {
            if (islands[i]->isStagnating(stagnatingIterations)) {
                int migrantsCount = 0.15 * islands[i]->population.size();
                if (islands[i]->islandType == EXPLOITATIVE) {                    
                    uniform_int_distribution<int> islandIndex(0, explorativeIslands.size() - 1);
                    size_t index = islandIndex(randomEngine);
                    migrateIsland(explorativeIslands[index], islands[i], migrantsCount);
                }
            }
        }

        Individual globalBest = findGlobalBest();
        applyFinalKMeans(globalBest);
        if (globalBest.calculateFitness(pointDistances) < bestIndividual.calculateFitness(pointDistances)) {
            bestIndividual = globalBest;
        }
        std::cout << "Iteration: " << iteration << ", best: " << bestIndividual.calculateFitness(pointDistances) << endl;
    }
}


void GeneticAlgorithm::migrateIsland(Island* source, Island* target, const int& migrantsCount) {
    vector<Individual> migrants;
    migrants.resize(migrantsCount);

    if (target->islandType == EXPLORATIVE) {
        // losowe zastepuja losowych
        uniform_int_distribution<int> distrPopulationIndex(0, source->population.size() - 1);
        for (size_t j = 0; j < migrantsCount; ++j) {
            migrants[j] = source->population[distrPopulationIndex(randomEngine)];
        }

        for (size_t j = 0; j < migrantsCount; ++j) {
            size_t randomIndex = distrPopulationIndex(randomEngine);
            target->population[randomIndex] = migrants[j];
        }
    }
    else {
        // najlepsze zastepuja najgorsze
        sort(source->population.begin(), source->population.end(),
            [](const Individual& a, const Individual& b) {
                return a.fitness < b.fitness;
            });

        for (size_t j = 0; j < migrantsCount; ++j) {
            migrants[j] = source->population[j];
        }


        sort(target->population.begin(),
            target->population.end(),
            [](const Individual& a, const Individual& b) {
                return a.fitness > b.fitness;
            });

        for (size_t j = 0; j < migrantsCount; ++j) {
            target->population[j] = migrants[j];
        }
    }
    
}

void GeneticAlgorithm::migrateIsland(const size_t& i, IslandType source, IslandType target, 
    const int& migrantsCount) {
    vector<Individual> migrants;
    migrants.resize(migrantsCount);

    Island* sourceIsland = islands[0];
    Island* targetIsland = islands[0];
    if (source == EXPLOITATIVE) {
        sourceIsland = exploitativeIslands[i];
        if (target == EXPLORATIVE) {
            uniform_int_distribution<int> distrIslandIndex(0, explorativeIslands.size() - 1);
            size_t nextExplorativeIslandIndex = distrIslandIndex(randomEngine);
            targetIsland = explorativeIslands[nextExplorativeIslandIndex];
        }
        else {
            uniform_int_distribution<int> distrIslandIndex(0, exploitativeIslands.size() - 1);
            size_t nextExploitativeIslandIndex = distrIslandIndex(randomEngine);
            while (nextExploitativeIslandIndex == i) {
                nextExploitativeIslandIndex = distrIslandIndex(randomEngine);
            }
            targetIsland = exploitativeIslands[nextExploitativeIslandIndex];
        }
    }
    else {
        sourceIsland = explorativeIslands[i];
        if (target == EXPLORATIVE) {
            uniform_int_distribution<int> distrIslandIndex(0, explorativeIslands.size() - 1);
            size_t nextExplorativeIslandIndex = distrIslandIndex(randomEngine);
            while (nextExplorativeIslandIndex == i) {
                nextExplorativeIslandIndex = distrIslandIndex(randomEngine);
            }
            targetIsland = explorativeIslands[nextExplorativeIslandIndex];
        }
        else {
            uniform_int_distribution<int> intDistr(0, exploitativeIslands.size() - 1);
            size_t nextExploitativeIslandIndex = intDistr(randomEngine);
            targetIsland = exploitativeIslands[nextExploitativeIslandIndex];
        }
    }
    migrateIsland(sourceIsland, targetIsland, migrantsCount);
}


void GeneticAlgorithm::performMigrationExploitative(const double& migratedPopulationPercent) {
    int migrantsCount = migratedPopulationPercent * exploitativeIslands[0]->population.size();
    for (size_t i = 0; i < exploitativeIslands.size(); ++i) {
        IslandType target;
        uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(randomEngine) <= 0.965) {
            target = EXPLORATIVE;
        }
        else {
            target = EXPLOITATIVE;
        }

        migrateIsland(i, EXPLOITATIVE, target, migrantsCount);
    }
}

void GeneticAlgorithm::performMigrationExplorative(const double& migratedPopulationPercent) {
    int migrantsCount = migratedPopulationPercent * explorativeIslands[0]->population.size();
    for (size_t i = 0; i < explorativeIslands.size(); ++i) {
        IslandType target;
        uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(randomEngine) <= 0.385) {
            target = EXPLORATIVE;
        }
        else {
            target = EXPLOITATIVE;
        }

        migrateIsland(i, EXPLORATIVE, target, migrantsCount);
    }
}

Individual GeneticAlgorithm::findGlobalBest() {
    Individual globalBest = islands[0]->bestIndividual;
    for (auto& island : islands) {
        if (island->bestIndividual.calculateFitness(pointDistances) < globalBest.calculateFitness(pointDistances)) {
            globalBest = island->bestIndividual;
        }
    }
    return globalBest;
}

void GeneticAlgorithm::applyFinalKMeans(Individual& bestIndividual) {
    KMeans kMeans(maxClusters, 1000, randomEngine, points);
    Individual improved = kMeans.getIndividual(*bestIndividual.getGenotype());

    if (improved.calculateFitness(pointDistances) < bestIndividual.calculateFitness(pointDistances)) {
        bestIndividual = improved;
    }
}


/*
void GeneticAlgorithm::performMigration(size_t migrantsCount) {
    for (size_t i = 0; i < islands.size(); ++i) {
        sort(islands[i].population.begin(), islands[i].population.end(),
            [](const Individual& a, const Individual& b) {
                return a.fitness < b.fitness;
            });

        vector<Individual> migrants(islands[i].population.begin(),
            islands[i].population.begin() + migrantsCount);

        ///////
        for (auto& migrant : migrants) {
            migrant.mutate(0.05, randomEngine, pointDistances);
        }
        ///////

        size_t nextIsland = (i + 1) % islands.size();
        sort(islands[nextIsland].population.begin(), islands[nextIsland].population.end(),
            [](const Individual& a, const Individual& b) {
                return a.fitness > b.fitness;
            });

        for (size_t j = 0; j < migrants.size(); ++j) {
            islands[nextIsland].population[islands[nextIsland].population.size() - 1 - j] = migrants[j];
        }
    }
}*/








///////////////////////

void GeneticAlgorithm::evaluatePopulation() {
    /*int bestIndex = 0;
    vector<future<double>> futures(populationSize);

    for (int i = 0; i < populationSize; ++i) {
        futures[i] = async(launch::async, [this, i]() {
            return population[i].calculateFitness();
            });
    }

    for (int i = 0; i < populationSize; ++i) {
        fitness[i] = futures[i].get();
    }

    bestIndex = min_element(fitness.begin(), fitness.end()) - fitness.begin();

    if (fitness[bestIndex] < bestIndividual.calculateFitness()) {
        bestIndividual = population[bestIndex];
        noImprovementCounter = 0;
    }
    else {
        noImprovementCounter++;
    }*/

    #pragma omp parallel for
    for (int i = 0; i < populationSize; ++i) {
        fitness[i] = population[i].calculateFitness(pointDistances);
    }
    int bestIndex = std::min_element(fitness.begin(), fitness.end()) - fitness.begin();

    #pragma omp critical
    {
        if (fitness[bestIndex] < bestIndividual.calculateFitness(pointDistances)) {
            bestIndividual = population[bestIndex];
            noImprovementCounter = 0;
        }
        else {
            noImprovementCounter++;
        }
    }
}


/*
void GeneticAlgorithm::run() {
    evaluatePopulation();

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        vector<Individual> newPopulation;

        vector<pair<double, size_t>> fitnessIndices;
        for (size_t i = 0; i < fitness.size(); ++i) {
            fitnessIndices.push_back({ fitness[i], i });
        }
        int eliteCount = static_cast<int>(eliteRate * populationSize);
        
        partial_sort(fitnessIndices.begin(), fitnessIndices.begin() + eliteCount, fitnessIndices.end(), 
            [](const pair<double, size_t>& p1, const pair<double, size_t>& p2) {
                return p1.first < p2.first;
            });

        double betterIndividualsChance = 0.0;
        double restIndividualsChance = 0.15;
        if (iteration > maxIterations / 2 && iteration % 15 == 0) {
            betterIndividualsChance = 0.35;
        }

        #pragma omp parallel for
        for (size_t i = 0; i < populationSize; ++i) {
            size_t index = fitnessIndices[i].second;
            double mutationChance = (i < eliteCount) ? betterIndividualsChance : restIndividualsChance;
            uniform_real_distribution<double> dist(0.0, 1.0);
            if (dist(randomEngine) < mutationChance) {
                population[index].adaptiveMutate(randomEngine, pointDistances);
            }

            if (i < eliteCount) {
                #pragma omp critical
                newPopulation.push_back(population[index]); 
            }
        }


        for (int i = eliteCount; i < populationSize; i += 2) {

            // jak z roulette
            Individual parent1, parent2;
            if (iteration < maxIterations / 3) {
                parent1 = tournamentSelection(6);
                parent2 = tournamentSelection(6);
            }
            else {
                parent1 = rouletteSelection();
                parent2 = rouletteSelection();
            }

            uniform_real_distribution<double> probability(0.0, 1.0);
            if (probability(randomEngine) < crossoverProbability) {
                if (iteration < maxIterations / 2) {
                    pair<Individual, Individual> r = parent1.onePointCrossover(parent2, randomEngine);
                    //pair<Individual, Individual> r = parent1.multiPointCrossover(parent2, randomEngine);
                    parent1 = move(r.first);
                    parent2 = move(r.second);
                }
                else {
                    pair<Individual, Individual> r = parent1.uniformCrossover(parent2, randomEngine);
                    parent1 = move(r.first);
                    parent2 = move(r.second);
                }
            }

            parent1.mutate(mutationProbability, randomEngine, pointDistances);
            parent2.mutate(mutationProbability, randomEngine, pointDistances);

            uniform_real_distribution<double> dist(0.0, 1.0);
            if (dist(randomEngine) < 0.1) {
                parent1.adaptiveMutate(randomEngine, pointDistances);
            }
            if (dist(randomEngine) < 0.1) {
                parent2.adaptiveMutate(randomEngine, pointDistances);
            }

            if (noImprovementCounter > improvementThreshold) {
                mutationProbability = min(1.0, mutationProbability + 0.05);
                KMeans kmeans(maxClusters, 1, randomEngine, points);
                parent1 = move(kmeans.getIndividual(*parent1.getGenotype()));
                parent2 = move(kmeans.getIndividual(*parent2.getGenotype()));

                if (parent1.calculateFitness(pointDistances) > bestIndividual.calculateFitness(pointDistances)
                    && parent2.calculateFitness(pointDistances) > bestIndividual.calculateFitness(pointDistances)) {
                    noImprovementCounter /= 2;
                }
                else {
                    noImprovementCounter = 0;
                }
            }

            newPopulation.push_back(parent1);
            newPopulation.push_back(parent2);
        }

        population = move(newPopulation);
        evaluatePopulation();

        std::cout << "Iteration " << iteration + 1 << " Best Fitness: " << bestIndividual.calculateFitness(pointDistances) << endl;

        if (iteration % 200 == 0 && iteration > 0) {
            //mutationProbability *= 0.95; // Reduce mutation rate over time
        }

    }

    KMeans kmeans(maxClusters, 30, randomEngine, points);
    Individual potential = kmeans.getIndividual(*bestIndividual.getGenotype());

    if (potential.calculateFitness(pointDistances) < bestIndividual.calculateFitness(pointDistances)) {
        bestIndividual = potential;
    }

    cout << bestIndividual.calculateFitness(pointDistances) << endl;
}



Individual& GeneticAlgorithm::tournamentSelection(int amountOfDraw) {
    int bestIndex = randomEngine() % populationSize;
    for (int i = 1; i < amountOfDraw; ++i) {
        int candidate = randomEngine() % populationSize;
        if (fitness[candidate] < fitness[bestIndex]) {
            bestIndex = candidate;
        }
    }
    return population[bestIndex];
}

Individual& GeneticAlgorithm::rouletteSelection() {
    double totalFitness = 0.0;
    for (double f : fitness) {
        totalFitness += f;
    }

    uniform_real_distribution<> dist(0, 1);
    double target = dist(randomEngine);
    double cumulative = 0.0;
    for (size_t i = 0; i < population.size(); ++i) {
        cumulative += fitness[i] / totalFitness;
        if (cumulative >= target) {
            return population[i];
        }
    }
    return population.back();
}

*/