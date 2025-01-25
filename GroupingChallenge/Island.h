#pragma once

#include "Individual.h"

enum IslandType { EXPLORATIVE, EXPLOITATIVE };

class Island {
public:
    Island(IslandType islandType, const int& populationSize, const double& mutationProb, const double& crossoverProb, 
        Individual&(Island::* selection)(), mt19937& randomEngine, const vector<vector<double>>& pointDistances,
        const vector<CPoint>& points, const int& maxClusters)
        : mutationProbability(mutationProb), crossoverProbability(crossoverProb), selectionMethod(selection), 
        randomEngine(randomEngine), maxClusters(maxClusters), islandType(islandType) {
        for (int i = 0; i < populationSize; ++i) {
            population.push_back(Individual(maxClusters, randomEngine, points));
        }
        fitness.resize(populationSize);
        this->pointDistances = pointDistances;
        this->points = points;
        evaluatePopulation();
    }

    
    void runGeneration(const size_t& iteration, const int& maxIterations);

    void runExploitative(const size_t& iteration, const int& maxIterations);
    void kMeansExploitative(const double& populationPercent);
    void kMeansForWorst(const double& populationPercent);

    void runExplorative(const size_t& iteration, const int& maxIterations);

    Individual& tournamentSelection();
    Individual& rouletteSelection();
    Individual& rouletteSelectionWorst();

    bool isStagnating(const size_t& noImprovementThreshold);


    vector<Individual> population;        // Populacja wyspy
    Individual bestIndividual;            // Najlepszy osobnik na wyspie
    IslandType islandType;
private:

    int noImprovementCounter = 0;         // Licznik stagnacji
    double mutationProbability;           // Prawdopodobieñstwo mutacji
    double crossoverProbability;          // Prawdopodobieñstwo krzy¿owania

    int maxFitnessIndex;

    vector<double> fitness;
    vector<vector<double>> pointDistances;

    Individual& (Island::* selectionMethod)();  // Metoda selekcji (turniejowa, ruletka)

    mt19937 randomEngine;

    int maxClusters;
    vector<CPoint> points;


    void evaluatePopulation();
    Individual& tournamentSelection(int amountOfDraw);

    double dynamicMutationRate(size_t iteration, size_t maxIterations);
    double dynamicCrossoverRate(size_t iteration, size_t maxIterations);

};


