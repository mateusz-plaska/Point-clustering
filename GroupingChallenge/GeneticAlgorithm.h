#pragma once

#include "Individual.h"
#include "Island.h"

#include <vector>
#include <random>
using namespace std;

class GeneticAlgorithm
{
public:
    GeneticAlgorithm(int populationSize, const vector<CPoint>& points, int maxClusters,
        double mutationProbability, double crossoverProbability,
        int maxIterations, mt19937& randomEngine);

    ~GeneticAlgorithm() {
        for (Island* island : islands) {
            delete island;
        }
    }

    void run();
    const Individual& getBestIndividual() const {
        return bestIndividual;
    }

    void initializePopulation();

private:
    
    void evaluatePopulation();



    vector<Individual> population;
    vector<double> fitness;
    Individual bestIndividual;

    vector<CPoint> points;

    int numberOfPoints;
    int maxClusters;

    int populationSize;
    double mutationProbability;
    double crossoverProbability;
    int maxIterations;
    double eliteRate = 0.05;

    int noImprovementCounter = 0;
    const int improvementThreshold = 60;

    mt19937& randomEngine;


    vector<vector<double>> pointDistances;
    void initializePointDistances();

    void migrateIsland(Island* source, Island* target, const int& migrantsCount);
    void migrateIsland(const size_t& islandIndex, IslandType source, IslandType target, const int& migrantsCount);
    void performMigrationExploitative(const double& migratedPopulationPercent);
    void performMigrationExplorative(const double& migratedPopulationPercent);



    vector<Island*> islands;
    vector<Island*> explorativeIslands;
    vector<Island*> exploitativeIslands;

    Individual findGlobalBest();
    void applyFinalKMeans(Individual& bestIndividual);
};

