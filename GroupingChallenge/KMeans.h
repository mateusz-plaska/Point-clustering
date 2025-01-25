#pragma once

#include "Individual.h"

#include <vector>
#include <random>
using namespace std;

class KMeans
{
public:
	KMeans(const int& maxClusters, const int maxIterations, mt19937& randomEngine, const vector<CPoint>& points)
		: maxClusters(maxClusters), maxIterations(maxIterations), randomEngine(randomEngine), points(points) {}

	Individual getIndividual();
	Individual getIndividual(const vector<int>& genotypeOriginal);

private:
	int maxClusters;
	int maxIterations;
	mt19937& randomEngine;
	vector<CPoint> points;

	void updateCentroids(vector<CPoint>& centroids, const vector<int>& genotype);
};

/*

Individual KMeans::getIndividual(const vector<int>& genotypeOriginal) {
	vector<int> genotype = genotypeOriginal;
	int numberOfPoints = points.size();

	vector<CPoint> centroids(maxClusters);
	for (int i = 0; i < maxClusters; ++i) {
		centroids[i] = points[i % numberOfPoints];
	}

	updateCentroids(centroids, genotype);

	for (int iteration = 0; iteration < maxIterations; ++iteration) {
		bool changed = false;

		#pragma omp parallel for reduction(|:changed)
		for (int i = 0; i < numberOfPoints; ++i) {
			double minDistance = numeric_limits<double>::max();
			int bestCluster = 0;

			for (int j = 0; j < maxClusters; ++j) {
				double distance = points[i].dCalculateDistance(centroids[j]);
				if (distance < minDistance) {
					minDistance = distance;
					bestCluster = j + 1;
				}
			}

			if (genotype[i] != bestCluster) {
				genotype[i] = bestCluster;
				changed = true;
			}
		}

		if (!changed) break;

		updateCentroids(centroids, genotype);
	}

	return Individual(genotype, maxClusters, points);
}

void KMeans::updateCentroids(vector<CPoint>& centroids, const vector<int>& genotype) {
	vector<CPoint> newCentroids(maxClusters);
	vector<int> clusterSizes(maxClusters, 0);

	#pragma omp parallel for
	for (int i = 0; i < points.size(); ++i) {
		int cluster = genotype[i];
		const vector<double>& pointCoordinates = points[i].getCoordinates();

		#pragma omp critical
		{
			vector<double>& newCentroidCoordinates = newCentroids[cluster - 1].getCoordinates();
			newCentroidCoordinates.resize(pointCoordinates.size());
			for (int dimension = 0; dimension < pointCoordinates.size(); ++dimension) {
				newCentroidCoordinates[dimension] += pointCoordinates[dimension];
			}
			++clusterSizes[cluster - 1];
		}
	}

	for (int i = 0; i < maxClusters; ++i) {
		if (clusterSizes[i] > 0) {
			vector<double>& newCentroidCoordinates = newCentroids[i].getCoordinates();
			for (int dimension = 0; dimension < newCentroidCoordinates.size(); ++dimension) {
				newCentroidCoordinates[dimension] /= clusterSizes[i];
			}
		}
		else {
			newCentroids[i] = centroids[i];
		}
	}

	centroids = move(newCentroids);
}

*/