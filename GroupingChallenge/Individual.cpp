#include "Individual.h"
#include "GroupingEvaluator.h"


#include <vector>
#include <random>
#include <iostream>
using namespace std;

Individual::Individual(const int& maxClusters, mt19937& randomEngine, const vector<CPoint>& points)
	: maxClusters(maxClusters), fitness(0.0), isFitnessCurrent(false), points(points) {
	numberOfPoints = points.size();
	uniform_int_distribution<int> distribution(1, maxClusters);
	genotype.resize(numberOfPoints);
	for (int& gene : genotype) {
		gene = distribution(randomEngine);
	}

}

Individual::Individual(const vector<int>& genotype, const int& maxClusters, const vector<CPoint>& points)
	: maxClusters(maxClusters), fitness(0.0), isFitnessCurrent(false), points(points) {
	numberOfPoints = points.size();
	this->genotype.resize(numberOfPoints);
	this->genotype = genotype;
}

void Individual::rebuildGroups() {
	groups.clear();
	groups.resize(maxClusters + 1);
	for (size_t i = 0; i < numberOfPoints; ++i) {
		groups[genotype[i]].push_back(i);
	}
}

double Individual::getPointDistance(const size_t& sourcePointIndex, const size_t& targetPointIndex, 
	const vector<vector<double>>& pointDistances) const {
	if (sourcePointIndex < targetPointIndex) {
		return pointDistances[sourcePointIndex][targetPointIndex - 1 - sourcePointIndex];
	}
	else if (targetPointIndex < sourcePointIndex) {
		return pointDistances[targetPointIndex][sourcePointIndex - 1 - targetPointIndex];
	}
	return 0.0;
}


double Individual::calculateFitness(const vector<vector<double>>& pointDistances) {
	if (!isFitnessCurrent) {
		rebuildGroups();

		double totalFitness = 0.0;

		#pragma omp parallel for reduction(+:totalFitness) schedule(dynamic)
		for (size_t clusterIndex = 1; clusterIndex <= maxClusters; ++clusterIndex) {
			const vector<size_t>& indices = groups[clusterIndex];
			for (size_t i = 0; i < indices.size() - 1; ++i) {
				for (size_t j = i + 1; j < indices.size(); ++j) {
					totalFitness += getPointDistance(indices[i], indices[j], pointDistances);
				}
			}
		}

		fitness = 2.0 * totalFitness;
		isFitnessCurrent = true;
	}
	return fitness;
}

double Individual::updateFitnessForGeneChange(const int& geneIndex, const int& oldCluster, const int& newCluster,
	const vector<vector<double>>& pointDistances) {
	if (!isFitnessCurrent) {
		return calculateFitness(pointDistances);
	}

	if (oldCluster == newCluster) {
		return fitness;
	}

	
	#pragma omp parallel for reduction(-:fitness) schedule(dynamic)
	for (size_t i = 0; i < numberOfPoints; ++i) {
		if (geneIndex != i) {
			if (genotype[i] == oldCluster) {
				fitness -= (2.0 * getPointDistance(geneIndex, i, pointDistances));
			}
			else if (genotype[i] == newCluster) {
				fitness += (2.0 * getPointDistance(geneIndex, i, pointDistances));
			}
		}
	}

	/*
	const auto& oldGroup = groups[oldCluster];
	const auto& newGroup = groups[newCluster];

	#pragma omp parallel for reduction(-:fitness) schedule(dynamic)
	for (size_t i = 0; i < oldGroup.size(); ++i) {
		if (oldGroup[i] != geneIndex) {
			fitness -= 2.0 * getPointDistance(geneIndex, i, pointDistances);
		}
	}
	
	#pragma omp parallel for reduction(+:fitness) schedule(dynamic)
	for (size_t i = 0; i < newGroup.size(); ++i) {
		if (newGroup[i] != geneIndex) {
			fitness += 2.0 * getPointDistance(geneIndex, i, pointDistances);
		}
	}

	groups[oldCluster].erase(remove(groups[oldCluster].begin(), groups[oldCluster].end(), geneIndex), 
		groups[oldCluster].end());
	groups[newCluster].push_back(geneIndex);*/


	return fitness;
}


pair<Individual, Individual> Individual::onePointCrossover(const Individual& other, mt19937& randomEngine) const {
	int pointIndex = randomEngine() % numberOfPoints;
	vector<int> genotype1 = this->genotype;
	vector<int> genotype2 = this->genotype;
	for (size_t i = 0; i < pointIndex; ++i) {
		genotype2[i] = other.genotype[i];
	}
	for (size_t i = pointIndex; i < numberOfPoints; ++i) {
		genotype1[i] = other.genotype[i];
	}
	return make_pair(Individual(move(genotype1), this->maxClusters, this->points), 
		Individual(move(genotype2), this->maxClusters, this->points));
}

pair<Individual, Individual> Individual::multiPointCrossover(Individual& other, mt19937& randomEngine) const {
	int pointIndex1 = randomEngine() % numberOfPoints;
	int pointIndex2 = randomEngine() % numberOfPoints;
	if (pointIndex1 > pointIndex2) {
		swap(pointIndex1, pointIndex2);
	}

	vector<int> genotype1 = this->genotype;
	vector<int> genotype2 = other.genotype;
	for (int i = pointIndex1; i <= pointIndex2; ++i) {
		genotype1[i] = other.genotype[i];
		genotype2[i] = this->genotype[i];
	}

	return make_pair(Individual(move(genotype1), this->maxClusters, this->points),
		Individual(move(genotype2), this->maxClusters, this->points));
}

pair<Individual, Individual> Individual::uniformCrossover(const Individual& other, mt19937& randomEngine) const {
	vector<int> genotype1 = this->genotype;
	vector<int> genotype2 = this->genotype;
	for (size_t i = 0; i < numberOfPoints; ++i) {
		if (randomEngine() % 2 == 0) {
			genotype1[i] = this->genotype[i];
		}
		else {
			genotype1[i] = other.genotype[i];
		}
		if (randomEngine() % 2 == 0) {
			genotype2[i] = this->genotype[i];
		}
		else {
			genotype2[i] = other.genotype[i];
		}
	}
	return make_pair(Individual(move(genotype1), this->maxClusters, this->points), 
		Individual(move(genotype2), this->maxClusters, this->points));
}

void Individual::mutate(double mutationProbability, mt19937& randomEngine, const vector<vector<double>>& pointDistances) {
	uniform_real_distribution<double> probability(0.0, 1.0);
	uniform_int_distribution<int> clusterDist(1, maxClusters);

	for (size_t i = 0; i < numberOfPoints; ++i) {
		if (probability(randomEngine) < mutationProbability) {
			int oldCluster = genotype[i];
			int newCluster = clusterDist(randomEngine);
			if (oldCluster != newCluster) {
				genotype[i] = newCluster;
				updateFitnessForGeneChange(i, oldCluster, newCluster, pointDistances);
			}
		}
	}
}

void Individual::adaptiveMutate(mt19937& randomEngine, const vector<vector<double>>& pointDistances) {
	uniform_int_distribution<int> clusterDist(1, maxClusters);

	calculateFitness(pointDistances);

	for (size_t i = 0; i < numberOfPoints; ++i) {
		double oldFitness = fitness;
		int oldCluster = genotype[i];
		int newCluster = clusterDist(randomEngine);
		if (oldCluster != newCluster) {
			genotype[i] = newCluster;
			updateFitnessForGeneChange(i, oldCluster, newCluster, pointDistances);
			if (oldFitness < fitness) {
				genotype[i] = oldCluster;
				updateFitnessForGeneChange(i, newCluster, oldCluster, pointDistances);
			}
		}
	}
}