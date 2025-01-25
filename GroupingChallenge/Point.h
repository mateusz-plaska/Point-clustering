#ifndef POINT_H
#define POINT_H

#include <cmath>
#include <vector>

using namespace std;

namespace NGroupingChallenge
{
	class CPoint
	{
	public:
		void vAddCoordinate(double dCoordinate);
		double dCalculateDistance(const CPoint& cOther) const;

		CPoint& operator=(const CPoint& other) {
			this->v_coordinates = other.v_coordinates;
			return *this;
		}

		vector<double>& getCoordinates() {
			return v_coordinates;
		}

	private:
		const double d_WRONG_DISTANCE_VALUE = -1;

		vector<double> v_coordinates;
	};
}

#endif