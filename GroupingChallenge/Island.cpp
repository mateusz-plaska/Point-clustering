#include "Island.h"
#include "KMeans.h"

#include <iostream>


void Island::runGeneration(const size_t& iteration, const int& maxIterations) {
    if (islandType == EXPLOITATIVE) {
        runExploitative(iteration, maxIterations);
    }
    else {
        runExplorative(iteration, maxIterations);
    }
}


void Island::runExploitative(const size_t& iteration, const int& maxIterations) {
    vector<Individual> newPopulation;

    ////////////////////////////

    double eliteRate = 0.05;
    vector<pair<double, size_t>> fitnessIndices;
    for (size_t i = 0; i < fitness.size(); ++i) {
        fitnessIndices.push_back({ fitness[i], i });
    }
    int eliteCount = static_cast<int>(eliteRate * population.size());
    if (eliteCount % 2) {
        eliteCount--;
    }

    partial_sort(fitnessIndices.begin(), fitnessIndices.begin() + eliteCount, fitnessIndices.end(),
        [](const pair<double, size_t>& p1, const pair<double, size_t>& p2) {
            return p1.first < p2.first;
        });

    double betterIndividualsChance = 0.0;
    double restIndividualsChance = 0.15;
    if (iteration % 100 == 0) {
        betterIndividualsChance = 0.3;
    }

    //#pragma omp parallel for
    //for (size_t i = 0; i < population.size(); ++i) {
    //    size_t index = fitnessIndices[i].second;
    //    double mutationChance = (i < eliteCount) ? betterIndividualsChance : restIndividualsChance;
    //    uniform_real_distribution<double> dist(0.0, 1.0);
    //    if (dist(randomEngine) < mutationChance) {
    //        population[index].adaptiveMutate(randomEngine, pointDistances);
    //    }

    //    if (i < eliteCount) {
    //        #pragma omp critical
    //        newPopulation.push_back(population[index]);
    //    }
    //}

    #pragma omp paraller for
    for (size_t i = 0; i < eliteCount; ++i) {
        #pragma omp critical
        newPopulation.push_back(population[fitnessIndices[i].second]);
    }


    int randomIndividualsCount = 0.04 * population.size();
    if (randomIndividualsCount % 2) {
        randomIndividualsCount--;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < randomIndividualsCount; ++i) {
        Individual randomIndividual = Individual(maxClusters, randomEngine, points);
        randomIndividual.calculateFitness(pointDistances);
        #pragma omp critical
        newPopulation.push_back(randomIndividual);
    }

    int endIndex = population.size() - randomIndividualsCount;



    #pragma omp parallel for
    for (size_t i = eliteCount; i < endIndex; i += 2) {
        Individual& parent1 = population[i];
        Individual& parent2 = population[i];

        uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(randomEngine) <= 0.35) {
            parent1 = this->rouletteSelection();
            parent2 = this->rouletteSelection();
        }
        else {
            parent1 = (this->*selectionMethod)();
            parent2 = (this->*selectionMethod)();
        }


        if (dist(randomEngine) < crossoverProbability) {
            auto offspring = parent1.multiPointCrossover(parent2, randomEngine);
            parent1 = move(offspring.first);
            parent2 = move(offspring.second);
        }

        parent1.mutate(mutationProbability, randomEngine, pointDistances);
        parent2.mutate(mutationProbability, randomEngine, pointDistances);


        double adaptiveProbability = min(0.4, mutationProbability + 
            (parent1.calculateFitness(pointDistances) / fitness[maxFitnessIndex]));
        if (dist(randomEngine) < adaptiveProbability) {
            parent1.adaptiveMutate(randomEngine, pointDistances);
        }
        adaptiveProbability = min(0.4, mutationProbability +
            (parent2.calculateFitness(pointDistances) / fitness[maxFitnessIndex]));
        if (dist(randomEngine) < adaptiveProbability) {
            parent2.adaptiveMutate(randomEngine, pointDistances);
        }

        #pragma omp critical
        {
            newPopulation.push_back(parent1);
            newPopulation.push_back(parent2);
        }
    }

    population = move(newPopulation);

    // kMeans
    if ((iteration + 1) % 60 == 0) {
        kMeansExploitative(0.2);
    }

    evaluatePopulation();

    std::cout << bestIndividual.calculateFitness(pointDistances) << endl;

}

void Island::kMeansExploitative(const double& populationPercent) {
    KMeans kMeans(maxClusters, 2, randomEngine, points);
    #pragma omp parallel for
    for (size_t i = 0; i < populationPercent * population.size(); ++i) {
        uniform_int_distribution<int> dist(0, population.size() - 1);
        size_t index = dist(randomEngine);
        Individual improved = kMeans.getIndividual(*population[index].getGenotype());
        if (improved.calculateFitness(pointDistances) < population[index].calculateFitness(pointDistances)) {
            #pragma omp critical
            population[index] = improved;
        }
    }
}

void Island::kMeansForWorst(const double& populationPercent) {
    KMeans kMeans(maxClusters, 2, randomEngine, points);
    #pragma omp parallel for
    for (size_t i = population.size() - populationPercent * population.size(); i < population.size(); ++i) {
        Individual improved = kMeans.getIndividual(*population[i].getGenotype());
        if (improved.calculateFitness(pointDistances) < population[i].calculateFitness(pointDistances)) {
            #pragma omp critical
            population[i] = improved;
        }
    }
}


void Island::runExplorative(const size_t& iteration, const int& maxIterations) {
    vector<Individual> newPopulation;
  
    int randomIndividualsCount = 0.15 * population.size();
    if (randomIndividualsCount % 2) {
        randomIndividualsCount++;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < randomIndividualsCount; ++i) {
        Individual randomIndividual = Individual(maxClusters, randomEngine, points);
        randomIndividual.calculateFitness(pointDistances);
        #pragma omp critical
        newPopulation.push_back(randomIndividual);
    }

    int endIndex = population.size() - randomIndividualsCount;

    #pragma omp parallel for
    for (size_t i = 0; i < endIndex; i += 2) {
        Individual& parent1 = population[i];
        Individual& parent2 = population[i];

        uniform_real_distribution<double> distSelectionMethod(0.0, 1.0);
        if (distSelectionMethod(randomEngine) < 0.39) {
            uniform_int_distribution<int> intDist(0, population.size() - 1);
            parent1 = population[intDist(randomEngine)];
            parent2 = population[intDist(randomEngine)];
        }
        else {
            parent1 = (this->*selectionMethod)();
            parent2 = (this->*selectionMethod)();
        }

        uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(randomEngine) < crossoverProbability) {
            auto offspring = parent1.uniformCrossover(parent2, randomEngine);
            parent1 = move(offspring.first);
            parent2 = move(offspring.second);
        }

        parent1.mutate(mutationProbability, randomEngine, pointDistances);
        parent2.mutate(mutationProbability, randomEngine, pointDistances);

        if (dist(randomEngine) < 0.17) { // 0.05
            parent1.adaptiveMutate(randomEngine, pointDistances);
        }
        if (dist(randomEngine) < 0.17) {
            parent2.adaptiveMutate(randomEngine, pointDistances);
        }

        #pragma omp critical
        {
            newPopulation.push_back(parent1);
            newPopulation.push_back(parent2);
        }
    }

    population = move(newPopulation);

    evaluatePopulation();

    std::cout << bestIndividual.calculateFitness(pointDistances) << endl;
}

bool Island::isStagnating(const size_t& noImprovementThreshold) {
    if (noImprovementCounter > noImprovementThreshold) {
        noImprovementCounter = 0;
        return true;
    }
    return false;
}



void Island::evaluatePopulation() {
    #pragma omp parallel for
    for (int i = 0; i < population.size(); ++i) {
        fitness[i] = population[i].calculateFitness(pointDistances);
    }
    int bestIndex = std::min_element(fitness.begin(), fitness.end()) - fitness.begin();

    maxFitnessIndex = std::max_element(fitness.begin(), fitness.end()) - fitness.begin();

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

Individual& Island::tournamentSelection() {
    return tournamentSelection(3);
}

Individual& Island::tournamentSelection(int amountOfDraw) {
    int bestIndex = randomEngine() % population.size();
    for (int i = 1; i < amountOfDraw; ++i) {
        int candidate = randomEngine() % population.size();
        if (fitness[candidate] < fitness[bestIndex]) {
            bestIndex = candidate;
        }
    }
    return population[bestIndex];
}

Individual& Island::rouletteSelection() {
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

Individual& Island::rouletteSelectionWorst() {
    double totalInvertedFitness = 0.0;
    for (double f : fitness) {
        totalInvertedFitness += 1.0 / (f + 1e-10);
    }

    uniform_real_distribution<> dist(0, 1);
    double target = dist(randomEngine);
    double cumulative = 0.0;
    for (size_t i = 0; i < population.size(); ++i) {
        double invertedFitness = 1.0 / (fitness[i] + 1e-10);
        cumulative += (invertedFitness / totalInvertedFitness);
        if (cumulative >= target) {
            return population[i];
        }
    }
    return population.back();
}



/*
    vector<Individual> newPopulation;

    ////////////////////////////

    double eliteRate = 0.05;
    vector<pair<double, size_t>> fitnessIndices;
    for (size_t i = 0; i < fitness.size(); ++i) {
        fitnessIndices.push_back({ fitness[i], i });
    }
    int eliteCount = static_cast<int>(eliteRate * population.size());

    partial_sort(fitnessIndices.begin(), fitnessIndices.begin() + eliteCount, fitnessIndices.end(),
        [](const pair<double, size_t>& p1, const pair<double, size_t>& p2) {
            return p1.first < p2.first;
        });

    double betterIndividualsChance = 0.0;
    double restIndividualsChance = 0.15;
    if (iteration > maxIterations / 2 && iteration % 15 == 0) {
        betterIndividualsChance = 0.3;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < population.size(); ++i) {
        size_t index = fitnessIndices[i].second;
        double mutationChance = (i < eliteCount) ? betterIndividualsChance : restIndividualsChance;
        uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(randomEngine) < mutationChance) {
            ///
            if (i < eliteCount && dist(randomEngine) < 0.38) {
                population[index].mutate(mutationProbability, randomEngine, pointDistances);
            }
            else {

                ///
                population[index].adaptiveMutate(randomEngine, pointDistances);
            }
        }

        if (i < eliteCount) {
            #pragma omp critical
            newPopulation.push_back(population[index]);
        }
    }


    ///////////////////////////////////
    int endIndex = population.size();
    if (iteration % 50 == 0) {
        endIndex -= 6;
    }

    ////

    #pragma omp parallel for
    for (size_t i = eliteCount; i < endIndex; i += 2) {
        Individual& parent1 = (this->*selectionMethod)();
        Individual& parent2 = (this->*selectionMethod)();

        // rozne rodzaje krzyzowan dla wyspy
        uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(randomEngine) < crossoverProbability) {
            auto offspring = parent1.onePointCrossover(parent2, randomEngine);
            parent1 = move(offspring.first);
            parent2 = move(offspring.second);
        }

        parent1.mutate(mutationProbability, randomEngine, pointDistances);
        parent2.mutate(mutationProbability, randomEngine, pointDistances);

        if (dist(randomEngine) < 0.1) {
            parent1.adaptiveMutate(randomEngine, pointDistances);
        }
        if (dist(randomEngine) < 0.1) {
            parent2.adaptiveMutate(randomEngine, pointDistances);
        }

        #pragma omp critical
        {
            newPopulation.push_back(parent1);
            newPopulation.push_back(parent2);
        }
    }

    /////
    if (iteration % 50 == 0) {
        #pragma omp parallel for
        for (size_t i = 0; i < 6; ++i) { // Dodaj 5 losowych osobników
            Individual randomIndividual = Individual(maxClusters, randomEngine, points);
            randomIndividual.calculateFitness(pointDistances);
            #pragma omp critical
            newPopulation.push_back(randomIndividual);
        }
    }/////

    population = move(newPopulation);

    evaluatePopulation();

    std::cout << bestIndividual.calculateFitness(pointDistances) << endl;


*/