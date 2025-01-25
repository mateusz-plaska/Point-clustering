#include "GaussianGroupingEvaluatorFactory.h"
#include "GroupingEvaluator.h"
#include "Optimizer.h"

#include "KMeans.h"

using namespace NGroupingChallenge;

int main()
{
	CGaussianGroupingEvaluatorFactory c_evaluator_factory(5, 100);

	c_evaluator_factory
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0)
		.cAddDimension(-100, 100, 1.0, 1.0);

	CGroupingEvaluator* pc_evaluator = c_evaluator_factory.pcCreateEvaluator(0);

	COptimizer c_optimizer(*pc_evaluator);

	c_optimizer.vInitialize();

	for (int i = 0; i < 1; i++)
	{
		c_optimizer.vRunIteration();
	}


	cout << "KONIEC - WYNIK EVALUATOREM: " << pc_evaluator->dEvaluate(c_optimizer.pvGetCurrentBest()) << endl;

	delete pc_evaluator;

	return 0;
}