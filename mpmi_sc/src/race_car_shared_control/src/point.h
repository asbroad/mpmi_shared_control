#ifndef POINT_RACE_CAR_H
#define POINT_RACE_CAR_H

#include <iostream>
using namespace std;

class Point
{

  public:
    double x;
    double y;

    Point(double x_in, double y_in) {
      x=x_in;
      y=y_in;
    }

};

ostream& operator<<(ostream& os, const Point& pt)
{
    os << pt.x << ", " << pt.y;
    return os;
}

#endif
