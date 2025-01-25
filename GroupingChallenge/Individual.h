#pragma once

#include "Point.h"

#include <vector>
#include <random>
#include <numeric>
using namespace std;
using namespace NGroupingChallenge;

class Individual 
{
public:
	Individual() : isFitnessCurrent(true), maxClusters(100), numberOfPoints(100) {
		numeric_limits<double> doubleLimits;
		fitness = doubleLimits.max();
	}
	Individual(const int& maxClusters, mt19937& randomEngine, const vector<CPoint>& points);
	Individual(const vector<int>& genotype, const int& maxClusters, const vector<CPoint>& points);

	pair<Individual, Individual> onePointCrossover(const Individual& other, mt19937& randomEngine) const;
	pair<Individual, Individual> multiPointCrossover(Individual& other, mt19937& randomEngine) const;
	pair<Individual, Individual> uniformCrossover(const Individual& other, mt19937& randomEngine) const;
	void mutate(double mutationProbability, mt19937& randomEngine, const vector<vector<double>>& pointDistances);
	void adaptiveMutate(mt19937& randomEngine, const vector<vector<double>>& pointDistances);

	double calculateFitness(const vector<vector<double>>& pointDistances);
	double updateFitnessForGeneChange(const int& geneIndex, const int& oldCluster, const int& newCluster, 
		const vector<vector<double>>& pointDistances);

	const vector<int>* getGenotype() const {
		return &genotype;
	}

	void setGenotype(const vector<int>& newGenotype) {
		genotype = newGenotype;
		isFitnessCurrent = false;
	}

	double fitness;

private:
	vector<int> genotype;
	vector<CPoint> points;
	int numberOfPoints;
	int maxClusters;
	//double fitness;
	bool isFitnessCurrent;
	vector<vector<size_t>> groups;

	void rebuildGroups();

	double getPointDistance(const size_t& sourcePointIndex, const size_t& targetPointIndex, const vector<vector<double>>& pointDistances) const;
};

