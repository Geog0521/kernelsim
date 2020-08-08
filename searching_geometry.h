
/** A class for data events and searching in 2D & 3D space.
* Author: Lingqing Yao
* Email: yaolingqing@gmail.com
* Date: 23 May, 2017
*      
**/

#ifndef __searching_geometry_h__
#define __searching_geometry_h__

#include <limits>
#include <algorithm>

#include <GsTLAppli/geostat/common.h>
#include <GsTLAppli/geostat/geostat_algo.h>
#include <GsTLAppli/utils/gstl_types.h>
#include <GsTLAppli/grid/grid_model/geovalue.h>
#include <GsTLAppli/grid/grid_model/property_copier.h> 
#include <GsTLAppli/grid/grid_model/sgrid_cursor.h> 
#include <GsTLAppli/grid/grid_model/rgrid.h>
#include <GsTLAppli/grid/grid_model/rgrid_neighborhood.h>


#include <vector>
#include <string>
#include <GsTLAppli/grid/grid_model/grid_region_temp_selector.h> 
#include <ctime>
#include "defs_type.h"
#include <cmath>
#include <cerrno>
#include <cfenv>

#pragma STDC FENV_ACCESS ON


/** The neighborhood class in SGEMS is fixed in the implementation of grid or point set data.
Fortunately, there is a base class "Search_filter" that we can derive and manipulate the
candidate points.

step 1: Sort the list of points in the neighborhood retrieved from the intial searching
in increasing order of distances from the center node.
step 2: For each point in the sorted list, determine the coding of octant where this point
locates in.

*/

class octant_2D_table
{
public:
	//
	octant_2D_table(double r, double rinc, double cell_size)
		:radius_(r),
		ring_inc_(rinc),
		cell_size_(cell_size)
	{

		//The first quandrant
		int dim = (int)(radius_ / cell_size_ + 0.5);
		int d = (2 * dim + 1) * (2 * dim + 1);
		table_.resize(d, -1);

		dim_ = dim;
		int i = 0;
		int ring = 0;
		for (int y = -dim; y <= dim; ++y) {
			for (int x = -dim; x <= dim; ++x)
			{

				double xabs = (double)(x >= 0 ? x : -x);
				double yabs = (double)(y >= 0 ? y : -y);

				double r = std::sqrt(xabs*xabs + yabs*yabs)*cell_size;
				if (r > radius_)
				{
					++i;
					continue;
				}

				ring = (int)(r / ring_inc_);

				int index = ring * 24;

				int zone = 0;

				const double pi = std::acos(-1);

				if (yabs < xabs * std::tan(pi / 12))
					zone = 0;
				else if (yabs < xabs * std::tan(pi / 6))
					zone = 1;
				else if (yabs < xabs)
					zone = 2;
				else if (yabs < xabs * std::tan(pi / 3))
					zone = 3;
				else if (yabs < xabs * std::tan(5 * pi / 12))
					zone = 4;
				else if (yabs >= xabs * std::tan(5 * pi / 12))
					zone = 5;
				else
					zone = -1;

				//Change the zone code according to different quadrants 
				if (x <= 0 && y > 0)
					zone = 11 - zone;
				else if (x < 0 && y <= 0)
					zone = 11 + zone;
				else if (x >= 0 && y < 0)
					zone = 23 - zone;
				else
					;

				if (xabs == 0 && yabs == 0)
					zone = -1;


				table_[i++] = index + zone;

			}
		}



	}

	//The function to tell which zone the location falls into.
	int sector(double x, double y)
	{
		if (x*x + y*y > radius_ *radius_)
			return -1;
		int i = (int)((x + radius_) / cell_size_);

		int j = (int)((y + radius_) / cell_size_);

		return table_[j*(2 * dim_ + 1) + i];

	}

	double radius()
	{
		return radius_;
	}



	friend std::ostream& operator<< (std::ostream &out, const octant_2D_table &tbl);

private:

	//The searching radius
	double radius_;
	//The increment of the rings, it should depend on the density of the sample data
	//For instance, 4% sample data means around 1 point in a 5*5 area.
	double ring_inc_;
	//
	double cell_size_;

	int dim_;

	//Storing the index of the sectors for each cell inside the circle.
	std::vector<int> table_;


};

typedef GsTLVector<GsTLDouble> GsTLGridVector;

//A slight structure to store the matching information
struct matching_info
{
	int visiting_id;
	int node_id;
	int matched_id;
	double matched_distance;
};


class octant_search_filter : public Search_filter
{
public:

	octant_search_filter()
	{
	}

	virtual int match(double x, double y, double z) { return 0; }
	virtual int match(int nid, GsTLGridVector& v, std::list<matching_info>& matchings) { return 0; }
	virtual double search_radius() = 0;

private:


};


class octant_2D_filter : public octant_search_filter
{
private:

	octant_2D_table oct_table_;
	std::vector<int> sectors_;
	std::vector<int> informed_sectors_;
	//Exclude points which are very close to the center node that can not be found from the sample data
	//This situation always happens after the simulation proceeds for half of the whole grid
	//that the conditioning data are nearby. 
	double radius_exclu_;

	int sequence_no_;
	
public:
	octant_2D_filter(double r, double rinc, double cell_size, double r_exl)
		:oct_table_(r, rinc, cell_size)
	{
		int ring = (int)(r/rinc + 0.5);
		sectors_.resize((ring+1) * 24, -1);

		radius_exclu_ = r_exl;

	}
    virtual ~octant_2D_filter(){}
    virtual bool is_admissible( const Geovalue& neigh, const Geovalue& center);

    virtual bool is_neighborhood_valid() {return true;}
    virtual void clear();
    virtual std::string class_name(){return "octant_2D_filter";} 
    virtual Search_filter* clone() {return new octant_2D_filter(*this);}

	int match(double x, double y, double z);
	double search_radius();
};


class search_filter : public octant_search_filter
{
private:

	//A vector to store the lags of the conditioning data
	std::vector<GsTLGridVector> geoms_;
	double lag_tol_;
	double band_tol_;
	double angle_tol_;

	//The searching radius
	double radius_;

	//Exclude points which are very close to the center node that can not be found from the sample data
	//This situation always happens after the simulation proceeds for half of the whole grid
	//that the conditioning data are nearby. 
	double radius_exclu_;

public:
	search_filter(double lag_tol, double band_tol, double angle_tol, double r, double r_exl) :
		lag_tol_(lag_tol),
		band_tol_(band_tol),
		angle_tol_(angle_tol),
		radius_(r),
		radius_exclu_(r_exl)
	{
	}

	virtual ~search_filter() {}
	virtual bool is_admissible(const Geovalue& neigh, const Geovalue& center);

	virtual bool is_neighborhood_valid() { return true; }
	virtual void clear();
	virtual std::string class_name() { return "search_filter"; }
	virtual Search_filter* clone() { return new search_filter(*this); }


	int match(int nid, int sz, GsTLGridVector& v, std::list<matching_info>& matchings);
	double search_radius() { return radius_; }

};




#endif