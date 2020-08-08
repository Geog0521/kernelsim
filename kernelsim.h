/** The main algorithm for kernel based simulation

* Author: Lingqing Yao
* Email: yaolingqing@gmail.com
* Date: June 4, 2017

*/


#ifndef __KERNEL_SIM_H_
#define __KERNEL_SIM_H_

#include <GsTLAppli/geostat/common.h>
#include <GsTLAppli/geostat/geostat_algo.h>
#include <GsTLAppli/utils/gstl_types.h>
#include <GsTLAppli/grid/grid_model/geovalue.h>
#include <GsTLAppli/grid/grid_model/property_copier.h> 
#include <GsTLAppli/grid/grid_model/sgrid_cursor.h> 
#include <GsTLAppli/grid/grid_model/rgrid.h>
#include <GsTLAppli/grid/grid_model/rgrid_neighborhood.h>

#include <GsTL/cdf/cdf_basics.h>
#include <GsTL/math/math_functions.h>
#include <GsTL/utils/gstl_error_messages.h>

#include <GsTLAppli/geostat/utilities.h> 
#include <GsTL/kriging/kriging_constraints.h> 
#include <GsTL/kriging/kriging_combiner.h> 

#include <cmath>
#include <boost/math/tools/minima.hpp>
#include <utility>


#include <vector>
#include <string>
#include <GsTLAppli/grid/grid_model/grid_region_temp_selector.h> 
#include <ctime>
#include "defs_type.h"


#include <GsTL/cdf/gaussian_cdf.h>
#include <GsTL/cdf_estimator/gaussian_cdf_Kestimator.h>
#include <GsTL/matrix_library/tnt/cholesky.h>

#include "QuadProg++.hh"
#include "searching_geometry.h"

#include <iostream>
#include <fstream>
#include  <iterator>

#undef inverse


#define STARTCLOCK()  start = clock()
#define ENDCLOCK() end = clock(); std::cout<<"The time is: "<<(end-start) <<" ms."<<std::endl

//Define a class to transform the original property into interval [-1,1]
class transform_to_legendre_domain
{
public:

	transform_to_legendre_domain(double zmin, double zmax)
		:zmin_(zmin),
		zmax_(zmax)
	{
		assert(zmin < zmax);
	}

	~transform_to_legendre_domain() {}

	template <typename T>
	void operator()(T val)
	{
		double z = val;

		val = (2 * z - zmin_ - zmax_) / (zmax_ - zmin_);

	}

private:

	double zmin_;
	double zmax_;
};

//Define a class to back transform the property on interval [-1,1] to the original scale

class back_from_legendre_domain
{
public:

	back_from_legendre_domain(double zmin, double zmax)
		:zmin_(zmin),
		zmax_(zmax)
	{
		assert(zmin < zmax);
	}

	template <typename  T>
	void operator()(T val)
	{
		double z = val;

		val = (z * (zmax_ - zmin_) + zmin_ + zmax_) / 2;
	}

private:

	double zmin_;
	double zmax_;
};


//We define a dummy class of cumulated probability function here from which the inversion
//of a probability will be given a meaningless value (no data value, GsTLGridProperty::no_data_value).
//This class is used to generate a null value for the simulating node if there is no conditional
//data in the neighborhood. There is no expectation to use this class elsewhere.
class Avg_dummy_cdf  //: public Cdf<double>
{
public:
	typedef GsTL::continuous_variable_tag   variable_category;

	Avg_dummy_cdf()
	{
		val_ = GsTLGridProperty::no_data_value;
	}

	void set_value(double val)
	{
		val_ = val;
	}
	virtual double inverse(double p) const { return val_; }
	virtual double lu_inverse(double p) const { return val_; }
	virtual double prob(double z) const { return 0; }

private:
	double val_;

};

//Some helpful functions related to the kernel space generated from Legendre polynomials (Lingqing Yao)

//The function to calculate the moment of Legendre polynomials of certain order w on the interval [-1, 1].
//For the simplicity, assuming now that the variance is unit 1.
//m is the mean and w is the order of the Legendre polynomial.
//int Legendre_Gauss_Mom(int max_order,  double m, double sigma, std::vector<double>& moments);
//Changed to include computation of derivatives for SGD. 2018-10-12
int Legendre_Gauss_Mom(int max_order,  double m, double sigma, std::vector<double>& moments, std::vector<double>& prototype_derivatives);
int Legendre_Gauss_Mom(int max_order,  double m, double sigma, std::vector<double>& moments);


//Define a new class of high-order simulation with the PDF approximation
//based on kernel methods. This class is derived from the base class
//<Geostat_algo> in SGEMS in order to act as a plugin in SGEMS if possible. 
class GEOSTAT_DECL kernelsim : public Geostat_algo {
public:

	kernelsim();
	~kernelsim();

	virtual bool initialize( const Parameters_handler* parameters,
		Error_messages_handler* errors );

	// The main procedure running the algorithm. 
	virtual int execute( GsTL_project* proj=0 );

	// Tells the name of the algorithm
	virtual std::string name() const { return "kernelsim"; }

public:
	//Create a instance of the algorithm
	static Named_interface* create_new_interface(std::string&);
	//clean the temporary properties or intermediate results
    void clean( GsTLGridProperty* prop = 0 );

private:
	//Initiate simulation grid
	bool get_simul_grid( const Parameters_handler* parameters,
		Error_messages_handler* errors );

	//Initiate the hard data
	bool get_hard_data( const Parameters_handler* parameters,
		Error_messages_handler* errors );
	
	//Initiate the training image
	bool get_training_image( const Parameters_handler* parameters,
		Error_messages_handler* error_mesgs );

	//Set up the covariance model
	bool set_up_covariance( const Parameters_handler* parameters,
		Error_messages_handler* errors );

	//Set up the neighborhoods
	bool set_up_neighborhood( const Parameters_handler* parameters,
		Error_messages_handler* errors );

	//Set up the regions
	bool set_up_regions( const Parameters_handler* parameters,
		Error_messages_handler* errors );

	// Set up the lower and upper bound values of the simulation
	bool set_bound_values( const Parameters_handler* parameters,
		Error_messages_handler* errors );

	//Build the kernel moments in the preprocessing step to save the computational time.
	//sigma is the kernel width, here Gaussian kernel is used currently.
	//2017-11-22
	bool build_kernel_moments( const Parameters_handler* parameters,
		Error_messages_handler* errors );

	//These functions are defined to implement the SGD optimization for the /sigma param. 31/08/2018
	void compute_derivative();
	void update_kernel_moments();
	//void update_prototype_moments();

	int optimize_kernel_width( );

	//Build a percentile table from the sample data for tranforming the data to [-1, 1] in proportion. Jan 10, 2019
	void build_frequency_table(double bin_width = 0.01);

	void build_frequency_table(std::vector<double>& cutoffs);


	bool build_legendre_hard_ti( const Parameters_handler* parameters,
		Error_messages_handler* errors );

private: 
	typedef Geostat_grid::location_type Location; 
	typedef Geostat_grid::property_type property_type;

	//Simulation grid
	Geostat_grid* simul_grid_;

	//To store the hard data in a Cartesian grid serving as the TI. 15/05/2019
	Geostat_grid* hard_data_ti_;


	MultiRealization_property* multireal_property_; 

	//Hard data grid
	Geostat_grid* harddata_grid_;   
	GsTLGridProperty* harddata_property_; 
	GsTLGridProperty* training_property_;
	GsTLGridProperty* hard_data_ti_property_;

	std::string harddata_property_name_; 
	//  Grid_initializer* initializer_; 
	SmartPtr<Property_copier> property_copier_;
	bool  assign_harddata_;

	//Training image
	std::string training_image_name_;
	std::string hard_ti_name_;
	std::string training_property_name_;
	RGrid* training_image_;


	//Temporary properties that were transformed from the original properties
	//into the interval [-1,1].
	GsTLGridProperty* temporary_harddata_property_;
	GsTLGridProperty* temporary_training_property_;

	//Neighborhood to find conditional data
	SmartPtr<Neighborhood> neighborhood_;

	//Hard data neighborhood
	SmartPtr<Neighborhood> hard_neigh_;

	long int seed_; 
	int nb_of_realizations_; 

	// Set up the regions
	Temporary_gridRegion_Selector grid_region_;
	Temporary_gridRegion_Selector hd_grid_region_;
	Temporary_gridRegion_Selector ti_grid_region_;

	//Define the boundary values including the min and max of z-values,
	//and the maximal order as well.
	int max_order_;
	property_type zmax_;
	property_type zmin_;
	property_type upperbound_;
	property_type lowerbound_;

	//March 2018 , the number of prototype distributions
	int num_prototypes_;

	int half_winx_;
	int half_winy_;
	int half_winz_;


	//number of replicates
	int num_ti_replicate_;
	int num_hd_replicate_;

	//searching tolerance
	double angle_tol_;
	double lag_tol_;
	double band_tol_;

	//Covariance model
    Covariance<Location> covar_;
    geostat_utils::KrigingCombiner* combiner_;
    geostat_utils::KrigingConstraints* Kconstraints_;


	//To store the kernel moments and Legendre polynomial values for the TI
	std::vector<GsTLGridProperty*> Legendre_moments_;
	std::vector<GsTLGridProperty*> Legendre_values_;

	std::vector<GsTLGridProperty*> Legendre_moments_hard_ti_;
	std::vector<GsTLGridProperty*> Legendre_values_hard_ti_;


	//kernel width
	double sigma_;

	int num_sel_protos_;

	int num_max_iterations_;
	double learning_rate_;
	bool b_optmize_width_;

	double sigma_lb_;
	double sigma_ub_;


	//Frequency and their corresponding cut-off value
	std::vector<double> freq_table_;
	std::vector<double> val_table_;
	double bin_width_;

};




struct basic_toleration
{
public:

	bool operator()(double x1, double x2)
	{
		return x2-x1 < 0.000001;
	}

};


//This base class provides a general way to inver the probability
//of a non-parametric distribution.
class Base_kernel_cdf //: public Cdf<double>
{
public:
	virtual double inverse(double p) const;
	virtual double prob(double z) const = 0;

private:
	class root_find_helper
	{
	public:
		root_find_helper(Base_kernel_cdf* ccdf, double p):
			ccdf_(ccdf),
			value_(p)
			{}
		~root_find_helper()
			{}
		double operator()(double x)
		{
			return ccdf_->prob(x) - value_;
		}

	private:
		double value_;
		Base_kernel_cdf* ccdf_;
	};

};


class Gaussian_kernel_cdf : public Base_kernel_cdf
{
public:
	virtual double prob(double z) const;


	//double inverse(double p) const {
	//	double x1 = Base_kernel_cdf::inverse(p); 
	//	Gaussian_cdf cdf(means_[0], variance_); 
	//	double x2= cdf.inverse(p); 
	//	return x2;}

	//Specialized functions for this kind of cdf
public:
	Gaussian_kernel_cdf()
		:Base_kernel_cdf(),
		learning_rate_(0.001),
		overall_shift_(0),
		overall_scaling_(1),
		balanced_(false)
	{
	}


	void learning_rate(double rate)
	{
		learning_rate_ = rate;
	}

	double learning_rate()
	{
		return learning_rate_;
	}

	void gradient(double g)
	{
		gradient_ = g;
	}

	double gradient()
	{
		return gradient_;
	}

	//
	void standard_deviation( double sigma) 
	{
		sigma_ = sigma;
		variance_ = sigma*sigma;
	}
	double standard_deviation() {return sigma_;}
	void means( const std::vector<double>& m) { means_ = m; }

	void denom( double d) {denom_ = d; }

	void clear() {
		means_.clear();
		coefs_.clear();
		shift_.clear();
		scaling_.clear();
		overall_shift_ = 0;
		overall_scaling_ = 1;
		balanced_ = false;
		//variance_ = 0;
	}
	void balance();
	void order(int o){max_order_ = o;}

	int order() {return max_order_;}

	//void num_prototypes(int n){num_prototypes_ = n;}

	int num_prototypes() {return means_.size();}

private:
	std::vector<double> shift_;
	std::vector<double> scaling_;
	double overall_shift_;
	double overall_scaling_;
	bool balanced_;
	//All the components share the same variance
	double variance_;
	double sigma_;
	//A vector of means corresponding to each component
	std::vector<double> means_;

	//The factor as the denominator dividing the sum of all the components
	double denom_;
	//
	int num_prototypes_;
	//ratio for each component
	std::vector<double> coefs_;

	int max_order_;
	double learning_rate_;
	double gradient_;
	

	//To facilitate the direct visit of the private members
	friend class replicate_processor;


};


class replicate_processor
{

	//Save all the replicates in the memory to compute the matrix for QuadProg later.
	std::vector<std::vector<int> > replicates_;
	//
	int nconditional_;
	//Number of prototypes used in the QP
	int num_prototypes_;
	//The indices of the selected prototypes from the predefined set.
	std::vector<int> validates_;

	//Now we need to build the matrix for the quadratic programming.
	QuadProg::Matrix<double> Q, Q_dup, CE, CI;
	QuadProg::Vector<double> q, q_dup, ce0, ci0, alpha;

	double gap_;

	//
	std::vector<double> coefs_pdf_;


public:


	typedef matrix_lib_traits< GSTL_TNT_lib > MatrixLib;


	//Develop the convolution operator of the conditioning data with the training image (it can either be the sample data or the TI)
	//Count the number of partial replicates of  the data event and compute their high-order statistics projected to the conditioning data space.
	//This function incoporate the kernel statistics both from the sample data and the ti.
	template<class GeovalueIterator,
		class GeovalueIterator2,
	  class Neighborhood
	>
	int kernel_convolution(GeovalueIterator begin, 
		GeovalueIterator end,
		GeovalueIterator2 ti_begin,
		GeovalueIterator2 ti_end,
		double inner_rad,
		const Neighborhood& neigh_cond,
		Gaussian_kernel_cdf & cdf,
		std::vector<GsTLGridProperty*>& Legendre_values,
		std::vector<GsTLGridProperty*>& Legendre_values_ti
		)
	  {

		typedef GeovalueIterator::value_type value_type;
		typedef GeovalueIterator::value_type::property_type property_type;
		typedef GeovalueIterator::value_type::location_type location_type;

		typedef Neighborhood::const_iterator const_iterator;

		typedef location_type::coordinate_type coordinate_type;

		typedef location_type::difference_type Euclidean_vector;

		//Get the training image grid from the iterator
		//Currently we only consider the TI is stored in a Regular Grid.
		const RGrid * cti = dynamic_cast<const RGrid*>( (*begin).grid());
		RGrid* ti = const_cast<RGrid*> (cti);
		assert(ti);

		cti = dynamic_cast<const RGrid*>((*ti_begin).grid());
		RGrid* ti2 = const_cast<RGrid*> (cti);
		assert(ti2);


		search_filter* sfilter =	dynamic_cast<search_filter*>
			(const_cast<Neighborhood&>(neigh_cond).search_neighborhood_filter() );
		assert(sfilter);
		//Get the searching radius
		double rad =  sfilter->search_radius();

		SmartPtr<Neighborhood> neigh_data = SmartPtr<Neighborhood>(ti->neighborhood(rad, rad, rad, 0, 0, 0) );

		neigh_data->max_size(50);

		static int nodes_sim = 0;
		static int hard_sim = 0;

		GeovalueIterator iter_ti = begin;

		int max_order = cdf.order();


		//Here is the code to retrieve the spatial template from the conditioning data (data event)
		value_type u = neigh_cond.center();

		int num_replicates = 0;

		int x, y, z;

		//(1)First, we need to generate a training data template from the visiting node and 
		//the conditional data within its neighborhood.

		nconditional_ = neigh_cond.size();
		std::vector<property_type> conditional(nconditional_);
		std::vector<Euclidean_vector> geometry_offsets(nconditional_);
		int i = 0;
		location_type center = u.location();

		std::vector<value_type> cond_vector(nconditional_);

		const double invalid_number = -99999;
		//Aslo store the center node in the replicate. 
		std::vector<int> replicate_vector(nconditional_+1, -1);


		for(const_iterator iter = neigh_cond.begin(); iter != neigh_cond.end(); ++ iter, ++ i){
		
			location_type loc = (*iter).location();
			conditional[i] = (*iter).property_value();

			geometry_offsets[i] = loc - center;

			//Copy the data event to a seperate vector
			cond_vector[i] = *iter;

		}
		
		double denom = 0.0;
		
		double var = 0.04;
		double diff = 0.0;	
		static int progressing = 0;


		///////////////////Compute the Legendre polynomial values of the conditioning data//////////////////////////////
		std::vector<std::vector<double>> Legendre_val_con(nconditional_);

		//Legendre polynomial values
		std::vector<double> P(max_order + 1, 0.0);

		for( int i = 0; i < nconditional_; ++ i)
		{

			P[0] = 1;
			P[1] = conditional[i];
			//The above are the initialized values for the recursive computations
			
			//Compute the Legendre polynomials of the conditioning data up to the maximum order
			//We don't store the first two orders' polynomials since they are trivial
			for(int w = 1; w < max_order; ++ w)
			{
				P[w+1] = (P[1]*P[w]*(2*w+1) - w*P[w-1])/double(w+1);
				Legendre_val_con[i].push_back(P[w+1]);
			}

		}
		////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//the number of the nodes inside a certain replicate partially matching the spatial template of the data event.
		int node_num = 0;
		////The vector to store the number of the partial replicates according to the number of the matching.
		std::vector<int> replicate_nums(nconditional_, 0);
		//Initializtion of the vector of coefficients of Legendre polynomials 
		std::vector<double> coefs_pdf(max_order+1, 0);
		std::vector<double> coefs_array(max_order+1, 0);
		coefs_pdf_.resize(max_order + 1);

		//compute kernel statistics from the TI
		int ti_rep_num = search_replicates(ti_begin, ti_end, neigh_cond);
		if (replicates_.size() >= 20)
		{
			nconditional_ = replicates_[0].size() - 1;
		}


		//std::map<int, std::pair<int, std::vector<double> > > partial_replicates;
		//The key is the template code, and the stored element is the replicates associated to this template code 2020-03-04
		std::vector<std::list<std::vector<int> > > partial_replicates(nconditional_);
		std::vector<std::map<int, std::list<std::vector<int> > > > replicates_list(nconditional_);


		//To visit the training image or hard data accoridng to a selected path
		for( ; iter_ti != end; ++ iter_ti)
		{
			//This is the number of the nodes of a partially matching replicate, which needs to be initialized to zero before each scan
			node_num = 0;

			diff = 0.0;
			int nid = (*iter_ti).node_id();
			//Changed to consider non-data-value OCT 25, 2018
			//Skip the location with no data value.
			if(!ti->selected_property()->is_informed(nid))
			{
				continue;
			}
			//get the location of the current visiting node
			ti->cursor()->coords( (*iter_ti).node_id(), x, y, z);

			//Get the value of the center node in the replicate
			double center_val = ti->selected_property()->get_value(nid);

			neigh_data->includes_center(false);

			//Get the neighborhood the current node on the sample grid
			neigh_data->find_neighbors(*iter_ti);

			i = 0;

			//Reintialize the vetor of replicate
			replicate_vector.assign(replicate_vector.size(), -1);
			//Store the matching information.
			std::list<matching_info> matchings;
			for(const_iterator iter = neigh_data->begin(); iter != neigh_data->end(); ++ iter, ++ i){

				GsTLGridVector dist = (*iter).location() - (*iter_ti).location();

				int ind = sfilter->match((*iter).node_id(), nconditional_, dist, matchings);

				//NOT match any point in the data event
				if(ind == 0)
					continue;

			}
			//Store the id of the center node. 
			replicate_vector[0] = nid;

			//Construct the replicate from the list of mathching information
			std::list<matching_info>::iterator it_mat = matchings.begin();
			while (!matchings.empty()) {

				it_mat = matchings.begin();
				int ind = (*it_mat).matched_id;
				int id = (*it_mat).node_id;

				replicate_vector[ind] = id;

				//Remove the matched index and node id from the matching list to avoid overlapping
				while (it_mat != matchings.end())
				{
					//it_mat = matchings.erase(it_mat);
					//if (it_mat == matchings.end())
					//	break;
					if (((*it_mat).matched_id == ind) )
						it_mat = matchings.erase(it_mat);
					else
						++it_mat;
					
				}

			}


			nid = -1;
			double coef_w = 0.0;

			double kew = 1.0;

			int template_index = 0;
			//The index 0 was reserved for the center node 
			for( int n = 1; n <= nconditional_; ++ n)
			{
				nid = replicate_vector[n];

				//Changed to consider non-data-value May 22, 2018
				if(nid == -1)
				{
					continue;
				}
				else{
					//the number of node matching to the spatial template increased by 1
					++ node_num;					

					//(n-1) here.
					template_index += 1<<(n-1);

				}

			}
			//When there is no common node between the replicate and the data event, just skip it
			if(node_num < 1)
				continue;

			//Up to this point, we found one partial replicate for the spatial template of the conditioning data
			replicates_list[node_num - 1][template_index].push_back(replicate_vector);
			
			//increase the number of the number of replicates with <node_number> nodes coincidental with the spatial configuration of the data event.
			++ replicate_nums[node_num - 1];
			//++ num_replicates;
		}

		int num_ensemble = 0;

		//Here we get the root node of a template DAG with highest numbers of matching nodes
		//and at the same time the number of replicates exceeds a certain statistical limit, say 20 here.
		std::map<int, std::list<std::vector<int> > >::iterator map_iter;

		int num_nodes_matching = nconditional_;
		int template_code = 0;
		int new_matching = 0;
		//Get the root node
		for (int n = num_nodes_matching - 1; n >= 0; --n)
		{
			map_iter =
				std::max_element(replicates_list[n].begin(), replicates_list[n].end(), [](const std::pair<int, std::list<std::vector<int> > >& a, const std::pair<int, std::list<std::vector<int> > >& b)->bool { return a.second.size() < b.second.size(); });

			if (map_iter != replicates_list[n].end())
			{
				if (map_iter->second.size() >= 20) {
					template_code = map_iter->first;
					new_matching = n + 1;
					break;
				}
			}
		}


		int matching_num = new_matching;

		std::vector<int> kernel_template(matching_num, -1);
		std::vector<int> inner_template(nconditional_-matching_num, -1);
		int nw = 0;

		if (matching_num >= 5)
		{
			int cnt = 0;
			int cnt2 = 0;
			int u = template_code;
			int i = 0;
			while (u)
			{
				if (u & 1)
				{
					kernel_template[cnt] = i;
					++cnt;
				}
				else
				{
					inner_template[cnt2] = i;
					++cnt2;
				}
				++i;
				u >>= 1;
			}
			//inner_template.resize(cnt2);
			while (i < nconditional_)
			{
				inner_template[cnt2] = i;
				++i;
				++cnt2;
			}



			int nstopping = matching_num;

			while (new_matching >= nstopping)
			{
				for (int n = num_nodes_matching - 1; n >= new_matching - 1; --n)
				{
					for (std::map<int, std::list<std::vector<int> > >::iterator  it_rep = replicates_list[n].begin();
						it_rep != replicates_list[n].end();
						)
					{
						int template_index = template_code & it_rep->first;
						if (template_index == template_code)
						{
							partial_replicates[new_matching - 1].splice(partial_replicates[new_matching - 1].end(), it_rep->second);
							std::map<int, std::list<std::vector<int> > >::iterator old_it = it_rep;
							++it_rep;
							replicates_list[n].erase(old_it);
						}
						else
						{
							++it_rep;
							continue;
						}
					}
				}


				template_code -= (1 << kernel_template[new_matching - 1]);
				--new_matching;
			}

			std::vector<std::vector<double> > kernel_stats(matching_num, std::vector<double>(max_order + 1, 0.0));
			

			//Compute the kernel statistics.
			for (int n = matching_num - 1; n >= nstopping - 1; --n)
			{
				//int n = matching_num - 1;
				for (std::list<std::vector<int> >::iterator it_par = partial_replicates[n].begin();
					it_par != partial_replicates[n].end();
					++it_par)
				{
					//First consider the center node.
					//The center node of the template scanning over the training image
					coefs_array[0] = 1;
					int nid = (*it_par)[0];
					double center_val = ti->selected_property()->get_value(nid);
					coefs_array[1] = center_val;
					for (int w = 2; w <= max_order; ++w)
					{
						coefs_array[w] = Legendre_values[w - 2]->get_value(nid);
					}

					for (int m = 0; m <= n; ++m)
					{
						int ind = kernel_template[m];
						nid = (*it_par)[ind + 1];
						//update the corresponding traning data and conditional data according to the position of
						//geometry template at the same time.
						double training_property = ti->selected_property()->get_value(nid);
						double condition_property = conditional[ind];

						//Update the coefficients
						double coef_w = 0.5 + 1.5 * training_property * condition_property;
						for (int w = 2; w <= max_order; ++w)
						{
							coef_w += (w + 0.5)
								* Legendre_values[w - 2]->get_value(nid)
								* Legendre_val_con[ind][w - 2];
						}
						for (int w = 0; w <= max_order; ++w) {

							//kew *= coef_w;
							coefs_array[w] *= coef_w;
							kernel_stats[m][w] += coefs_array[w];
							if (m + 1 < matching_num && m > nstopping - 1)
								kernel_stats[m + 1][w] -= coefs_array[w];
						}


					}
				}
			}

			nw = 0;


			for (int n = matching_num - 1; n >= nstopping - 1; --n)
			{
				//int n = matching_num - 1;
				//The number of replicates with n matching nodes
				nw += partial_replicates[n].size();
				for (int w = 0; w <= max_order; ++w) {

					coefs_pdf[w] += kernel_stats[n][w] / nw;
				}

			}
		}
		else {

			kernel_template.resize(0);
			inner_template.resize(nconditional_);
			for (int i = 0; i < inner_template.size(); ++i)
			{
				inner_template[i] = i;
			}
		}

		if (inner_template.size() > 0) {
			std::vector<double> ker_stats_ti(max_order + 1, 0.0);
			std::vector<double> inner_ker_stats_ti(max_order + 1, 0.0);

		num_replicates = replicates_.size();
		//Going through all the replicates we've found
		for (int i = 0; i < num_replicates; ++i)
		{
			//The center node of the template scanning over the training image
			coefs_array[0] = 1;
			coefs_array[1] = ti2->selected_property()->get_value(replicates_[i][0]);

			for (int w = 2; w <= max_order; ++w)
			{
				coefs_array[w] = Legendre_values_ti[w - 2]->get_value(replicates_[i][0]);
			}


			//For the rest of nodes in the replicate corresponding to the conditioning data.
			double coef_w;

			if (kernel_template.size() > 0) {
				//Consider the outer template first.
				for (int m = 0; m < kernel_template.size(); ++m)
				{
					int ind = kernel_template[m];
					int nid = replicates_[i][ind + 1];
					//update the corresponding traning data and conditional data according to the position of
					//geometry template at the same time.
					double training_property = ti2->selected_property()->get_value(nid);
					double condition_property = conditional[ind];
					

					//Update the coefficients
					coef_w = 0.5 + 1.5 * training_property * condition_property;
					for (int w = 2; w <= max_order; ++w)
					{
						coef_w += (w + 0.5)
							* Legendre_values_ti[w - 2]->get_value(nid)
							* Legendre_val_con[ind][w - 2];
					}
					for (int w = 0; w <= max_order; ++w) {
						coefs_array[w] *= coef_w;
					}

				}
				for (int w = 0; w <= max_order; ++w) {

					//ker_stats_ti[w] -= coefs_array[w];
					inner_ker_stats_ti[w] += coefs_array[w];

					
				}
			}


				//Continue with the inner template
				for (int m = 0; m < inner_template.size(); ++m)
				{
					int ind = inner_template[m];
					int nid = replicates_[i][ind + 1];

					double training_property = ti2->selected_property()->get_value(nid);
					double condition_property = conditional[ind];

					//Update the coefficients
					coef_w = 0.5 + 1.5 * training_property * condition_property;

					for (int w = 2; w <= max_order; ++w)
					{
						coef_w += (w + 0.5)
							* Legendre_values_ti[w - 2]->get_value(nid)
							* Legendre_val_con[ind][w - 2];
					}
					for (int w = 0; w <= max_order; ++w) {
						coefs_array[w] *= coef_w;
						//ker_stats_ti[w] += coefs_array[w];
					}
				}
				for (int w = 0; w <= max_order; ++w) {
					ker_stats_ti[w] += coefs_array[w];
				}

		}

			for (int w = 0; w <= max_order; ++w) {
			
				coefs_pdf[w] += (ker_stats_ti[w]-inner_ker_stats_ti[w]) / num_replicates;
			}
			++hard_sim;
		}


		for (int w = 1; w <= max_order; ++w) {
			coefs_pdf_[w] = coefs_pdf[w] * (w + 0.5) / coefs_pdf[0];
		}
		coefs_pdf_[0] = 0.5;

		num_replicates = nw+ ti_rep_num;

		sfilter->clear();

		//std::cout << "The current simulaiton node is: "<< ++ nodes_sim <<" hard data sim: " << hard_sim << " Number of replicates: " << num_replicates <<" Inner size: " << inner_template.size() << "  outer size: " << kernel_template.size() << std::endl;
		return num_replicates;
	  }

	  template<class GeovalueIterator,
		  class Neighborhood
	  >
		  int kernel_convolution(GeovalueIterator begin,
			  GeovalueIterator end,
			  const Neighborhood& neigh_cond,
			  Gaussian_kernel_cdf & cdf,
			  std::vector<GsTLGridProperty*>& Legendre_values
		  )
	  {

		  typedef GeovalueIterator::value_type value_type;
		  typedef GeovalueIterator::value_type::property_type property_type;
		  typedef GeovalueIterator::value_type::location_type location_type;

		  typedef Neighborhood::const_iterator const_iterator;

		  typedef location_type::coordinate_type coordinate_type;

		  typedef location_type::difference_type Euclidean_vector;

		  //Get the training image grid from the iterator
		  //Currently we only consider the TI is stored in a Regular Grid.
		  const RGrid * cti = dynamic_cast<const RGrid*>((*begin).grid());
		  RGrid* ti = const_cast<RGrid*> (cti);
		  assert(ti);

		  search_filter* sfilter = dynamic_cast<search_filter*>
			  (const_cast<Neighborhood&>(neigh_cond).search_neighborhood_filter());
		  assert(sfilter);
		  //Get the searching radius
		  double rad = sfilter->search_radius();

		  SmartPtr<Neighborhood> neigh_data = SmartPtr<Neighborhood>(ti->neighborhood(rad, rad, rad, 0, 0, 0));

		  neigh_data->max_size(50);

		  static int nodes_sim = 0;

		  GeovalueIterator iter_ti = begin;

		  int max_order = cdf.order();


		  //Here is the code to retrieve the spatial template from the conditioning data (data event)
		  value_type u = neigh_cond.center();

		  int num_replicates = 0;

		  int x, y, z;

		  //(1)First, we need to generate a training data template from the visiting node and 
		  //the conditional data within its neighborhood.

		  nconditional_ = neigh_cond.size();
		  std::vector<property_type> conditional(nconditional_);
		  std::vector<Euclidean_vector> geometry_offsets(nconditional_);
		  int i = 0;
		  location_type center = u.location();

		  std::vector<value_type> cond_vector(nconditional_);

		  const double invalid_number = -99999;
		  //Aslo store the center node in the replicate. 2020-03-22
		  std::vector<int> replicate_vector(nconditional_ + 1, -1);

		  for (const_iterator iter = neigh_cond.begin(); iter != neigh_cond.end(); ++iter, ++i) {

			  location_type loc = (*iter).location();
			  conditional[i] = (*iter).property_value();

			  geometry_offsets[i] = loc - center;

			  //Copy the data event to a seperate vector
			  cond_vector[i] = *iter;

		  }

		  static int progressing = 0;


		  ///////////////////Compute the Legendre polynomial values of the conditioning data//////////////////////////////
		  std::vector<std::vector<double>> Legendre_val_con(nconditional_);

		  //Legendre polynomial values
		  std::vector<double> P(max_order + 1, 0.0);

		  for (int i = 0; i < nconditional_; ++i)
		  {

			  P[0] = 1;
			  P[1] = conditional[i];
			  //The above are the initialized values for the recursive computations

			  //Compute the Legendre polynomials of the conditioning data up to the maximum order
			  //We don't store the first two orders' polynomials since they are trivial
			  for (int w = 1; w < max_order; ++w)
			  {
				  P[w + 1] = (P[1] * P[w] * (2 * w + 1) - w*P[w - 1]) / double(w + 1);
				  Legendre_val_con[i].push_back(P[w + 1]);
			  }

		  }
		  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
		  //the number of the nodes inside a certain replicate partially matching the spatial template of the data event.
		  int node_num = 0;
		  ////The vector to store the number of the partial replicates according to the number of the matching.
		  std::vector<int> replicate_nums(nconditional_, 0);
		  //Initializtion of the vector of coefficients of Legendre polynomials 
		  std::vector<double> coefs_pdf(max_order + 1, 0);
		  std::vector<double> coefs_array(max_order + 1, 0);
		  coefs_pdf_.resize(max_order + 1);


		  //std::map<int, std::pair<int, std::vector<double> > > partial_replicates;
		  //The key is the template code, and the stored element is the replicates associated to this template code 2020-03-04
		  std::vector<std::list<std::vector<int> > > partial_replicates(nconditional_);
		  std::vector<std::map<int, std::list<std::vector<int> > > > replicates_list(nconditional_);


		  //To visit the training image or hard data accoridng to a selected path
		  for (; iter_ti != end; ++iter_ti)
		  {
			  //This is the number of the nodes of a partially matching replicate, which needs to be initialized to zero before each scan
			  node_num = 0;

			  int nid = (*iter_ti).node_id();
			  //Changed to consider non-data-value OCT 25, 2018
			  //Skip the location with no data value.
			  if (!ti->selected_property()->is_informed(nid))
			  {
				  continue;
			  }
			  //get the location of the current visiting node
			  ti->cursor()->coords((*iter_ti).node_id(), x, y, z);

			  //Get the value of the center node in the replicate
			  double center_val = ti->selected_property()->get_value(nid);

			  //neigh_data->includes_center(false);

			  //Get the neighborhood the current node on the sample grid
			  neigh_data->find_neighbors(*iter_ti);

			  i = 0;

			  //Reintialize the vetor of replicate
			  replicate_vector.assign(replicate_vector.size(), -1);
			  //Store the matching information.
			  std::list<matching_info> matchings;
			  for (const_iterator iter = neigh_data->begin(); iter != neigh_data->end(); ++iter, ++i) {

				  GsTLGridVector dist = (*iter).location() - (*iter_ti).location();
				  //int ind = sfilter->match(dist.x(), dist.y(), dist.z());
				  int ind = sfilter->match((*iter).node_id(), nconditional_, dist, matchings);

				  //NOT match any point in the data event
				  if (ind == 0)
					  continue;

			  }
			  //Store the id of the center node. 
			  replicate_vector[0] = nid;

			  //Construct the replicate from the list of mathching information
			  std::list<matching_info>::iterator it_mat = matchings.begin();
			  while (!matchings.empty()) {

				  it_mat = matchings.begin();
				  int ind = (*it_mat).matched_id;
				  int id = (*it_mat).node_id;

				  replicate_vector[ind] = id;

				  //Remove the matched index and node id from the matching list to avoid overlapping
				  while (it_mat != matchings.end())
				  {
					  if (((*it_mat).matched_id == ind))
						  it_mat = matchings.erase(it_mat);
					  else
						  ++it_mat;

				  }

			  }


			  nid = -1;
			  double coef_w = 0.0;

			  double kew = 1.0;

			  int template_index = 0;
			  //The index 0 was reserved for the center node 2020-03-22
			  for (int n = 1; n <= nconditional_; ++n)
			  {
				  nid = replicate_vector[n];

				  //Changed to consider non-data-value May 22, 2018
				  if (nid == -1)
				  {
					  continue;
				  }
				  else {
					  //the number of node matching to the spatial template increased by 1
					  ++node_num;

					  //(n-1) here.
					  template_index += 1 << (n - 1);

				  }

			  }
			  //When there is no common node between the replicate and the data event, just skip it
			  if (node_num < 1)
				  continue;


			  //Up to this point, we found one partial replicate for the spatial template of the conditioning data
			  replicates_list[node_num - 1][template_index].push_back(replicate_vector);

			  //increase the number of the number of replicates with <node_number> nodes coincidental with the spatial configuration of the data event.
			  ++replicate_nums[node_num - 1];
			  //++ num_replicates;
		  }

		  int num_ensemble = 0;

		  //Here we get the root node of a template DAG with highest numbers of matching nodes
		  //and at the same time the number of replicates exceeds a certain statistical limit, say 20 here.
		  std::map<int, std::list<std::vector<int> > >::iterator map_iter;

		  int num_nodes_matching = nconditional_;
		  int template_code = 0;
		  int new_matching = 0;
		  //Get the root node
		  for (int n = num_nodes_matching - 1; n >= 0; --n)
		  {
			  map_iter =
				  std::max_element(replicates_list[n].begin(), replicates_list[n].end(), [](const std::pair<int, std::list<std::vector<int> > >& a, const std::pair<int, std::list<std::vector<int> > >& b)->bool { return a.second.size() < b.second.size(); });

			  if (map_iter != replicates_list[n].end())
			  {
				  if (map_iter->second.size() >= 20) {
					  template_code = map_iter->first;
					  new_matching = n + 1;
					  break;
				  }
			  }
		  }

		  //In case that no root node is found
		  if (template_code == 0)
		  {
			  std::cout << "The current simulaiton node is: " << ++nodes_sim << "BAD POINT" << std::endl;

			  return 0;
		  }

		  int matching_num = new_matching;

		  std::vector<int> kernel_template(matching_num, -1);

		  {
			  int cnt = 0;
			  int u = template_code;
			  int i = 0;
			  while (u)
			  {
				  if (u & 1)
				  {
					  kernel_template[cnt] = i;
					  ++cnt;
				  }
				  ++i;
				  u >>= 1;
			  }
		  }

		  int nstopping = matching_num;// 3;

		  while (new_matching >= nstopping)
		  {
			  for (int n = num_nodes_matching - 1; n >= new_matching - 1; --n)
			  {
				  for (std::map<int, std::list<std::vector<int> > >::iterator  it_rep = replicates_list[n].begin();
					  it_rep != replicates_list[n].end();
					  )
				  {
					  int template_index = template_code & it_rep->first;
					  if (template_index == template_code)
					  {
						  partial_replicates[new_matching - 1].splice(partial_replicates[new_matching - 1].end(), it_rep->second);
						  std::map<int, std::list<std::vector<int> > >::iterator old_it = it_rep;
						  ++it_rep;
						  replicates_list[n].erase(old_it);
					  }
					  else
					  {
						  ++it_rep;
						  continue;
					  }
				  }
			  }


			  template_code -= (1 << kernel_template[new_matching - 1]);
			  --new_matching;
		  }

		  std::vector<std::vector<double> > kernel_stats(matching_num, std::vector<double>(max_order + 1, 0.0));


		  //Compute the kernel statistics.
		  for (int n = matching_num - 1; n >= nstopping - 1; --n)
		  {
			  //int n = matching_num - 1;
			  for (std::list<std::vector<int> >::iterator it_par = partial_replicates[n].begin();
				  it_par != partial_replicates[n].end();
				  ++it_par)
			  {
				  //First consider the center node.
				  //The center node of the template scanning over the training image
				  coefs_array[0] = 1;
				  int nid = (*it_par)[0];
				  double center_val = ti->selected_property()->get_value(nid);
				  coefs_array[1] = center_val;
				  for (int w = 2; w <= max_order; ++w)
				  {
					  coefs_array[w] = Legendre_values[w - 2]->get_value(nid);
				  }

				  for (int m = 0; m <= n; ++m)
				  {
					  int ind = kernel_template[m];
					  nid = (*it_par)[ind + 1];
					  //update the corresponding traning data and conditional data according to the position of
					  //geometry template at the same time.
					  double training_property = ti->selected_property()->get_value(nid);
					  double condition_property = conditional[ind];

					  //Update the coefficients
					  double coef_w = 0.5 + 1.5 * training_property * condition_property;
					  for (int w = 2; w <= max_order; ++w)
					  {
						  coef_w += (w + 0.5)
							  * Legendre_values[w - 2]->get_value(nid)
							  * Legendre_val_con[ind][w - 2];
					  }
					  for (int w = 0; w <= max_order; ++w) {

						  //kew *= coef_w;
						  coefs_array[w] *= coef_w;
						  kernel_stats[m][w] += coefs_array[w];
						  if (m + 1 < matching_num && m > nstopping - 1)
							  kernel_stats[m + 1][w] -= coefs_array[w];
					  }


				  }
			  }
		  }

		  int nw = 0;


		  for (int n = matching_num - 1; n >= nstopping - 1; --n)
		  {
			  //int n = matching_num - 1;
			  //The number of replicates with n matching nodes
			  nw += partial_replicates[n].size();
			  for (int w = 0; w <= max_order; ++w) {

				  coefs_pdf[w] += kernel_stats[n][w] / nw;
			  }

		  }
		  for (int w = 1; w <= max_order; ++w) {
			  coefs_pdf_[w] = coefs_pdf[w] * (w + 0.5) / coefs_pdf[0];
		  }
		  coefs_pdf_[0] = 0.5;

		  num_replicates = nw;
		  num_replicates = partial_replicates[matching_num - 1].size();
		  		  		


		  sfilter->clear();

		  //std::cout << "The current simulaiton node is: " << ++nodes_sim << "Number of replicates: " << num_replicates << " matching num: "<< matching_num<<std::endl;

		  return num_replicates;
	  }

	//Write separate functions for different steps in the whole processing. 2018-10-31
	//This functions serves to search the replicates from either the TI or the hard data.

	template<class GeovalueIterator,
	  class Neighborhood
	>
	int search_replicates(GeovalueIterator begin, 
		GeovalueIterator end,
		const Neighborhood& neigh_cond
		)
	{

		static int nodes = 0;

		typedef GeovalueIterator::value_type::property_type property_type;
		typedef GeovalueIterator::value_type::location_type location_type;

		typedef Neighborhood::const_iterator const_iterator;
		typedef Neighborhood::value_type value_type;

		typedef location_type::coordinate_type coordinate_type;

		typedef GsTLGridNode::difference_type Euclidean_vector;


		value_type u = neigh_cond.center();

		int num_replicates = 0;

		//(1)First, we need to generate a training data template from the visiting node and 
		//the conditional data within its neighborhood.

		nconditional_ = neigh_cond.size();
		std::vector<property_type> conditional(nconditional_);
		std::vector<int> geometry_offsets(nconditional_*location_type::dimension);

		//Get the training image grid from the iterator
		const RGrid * ti = dynamic_cast<const RGrid*>( (*begin).grid());
		assert(ti);


		int i = 0;
		location_type center = u.location();

		for(const_iterator iter = neigh_cond.begin(); iter != neigh_cond.end(); ++ iter, ++ i){
		
			location_type loc = (*iter).location();
			conditional[i] = (*iter).property_value();
			for(int j = 0; j < location_type::dimension; ++ j){

				geometry_offsets[i*location_type::dimension + j] = GsTL::floor((loc[j] - center[j])/(ti->geometry()->cell_dims())[j]);
			}

		}

		std::vector<int> replicate(nconditional_+1);

		int x, y, z;		
		
		{

			num_replicates = 0;
			replicates_.clear();


			GeovalueIterator iter_ti = begin;

			//To visit the training image or hard data accoridng to a selected path
			for( ; iter_ti != end; ++ iter_ti)
			{

				int nid = (*iter_ti).node_id();
				//Changed to consider non-data-value OCT 25, 2018
				if(!ti->selected_property()->is_informed(nid))
				{
					continue;
				}

				ti->cursor()->coords( (*iter_ti).node_id(), x, y, z);


				replicate[0] = nid;
				//replicate[0] = (*iter_ti).node_id();
				for( i = 0; i < nconditional_; ++ i)
				{
					nid = ti->cursor()->node_id(x + geometry_offsets[i*location_type::dimension],
						y + geometry_offsets[i*location_type::dimension + 1],
						z + geometry_offsets[i*location_type::dimension + 2]);
					//Changed to consider non-data-value May 22, 2018
					if(!ti->selected_property()->is_informed(nid))
					{
						nid = -1;
						break;
					}
					else{
						replicate[i+1] = nid;

					}
				}
				if(nid == -1)
					continue;

				//save the replicate.
				replicates_.push_back(replicate);

				++ num_replicates;
			}

		}

		cout << "The simulation node is: NO. "<< ++ nodes  << "\t\t number of replicates:\t" <<num_replicates << "\t\t conditioning data:\t"<< nconditional_ << endl;

		return num_replicates;
}



	//Write separate functions for different steps in the whole processing. 2018-10-31
	//This functions serves to fit the high-order statistics of the replicates by Legendre polynomials.

	template<class GeovalueIterator,
	  class Neighborhood
	>
	int fit_replicates(GeovalueIterator begin, 
		GeovalueIterator end,
		const Neighborhood& neigh_cond,
		Gaussian_kernel_cdf & cdf,
		std::vector<GsTLGridProperty*>& Legendre_values
		)
	  {

		typedef GeovalueIterator::value_type::property_type property_type;
		typedef GeovalueIterator::value_type::location_type location_type;

		typedef Neighborhood::const_iterator const_iterator;

		typedef location_type::coordinate_type coordinate_type;

		typedef GsTLGridNode::difference_type Euclidean_vector;

		//Get the training image grid from the iterator
		const RGrid * ti = dynamic_cast<const RGrid*>( (*begin).grid());
		assert(ti);


		int max_order = cdf.order();
		///////////////////Compute the Legendre polynomial values of the conditioning data//////////////////////////////
		std::vector<property_type> conditional(neigh_cond.size());

		int i = 0;
		for(const_iterator iter = neigh_cond.begin(); iter != neigh_cond.end(); ++ iter, ++ i){		
			conditional[i] = (*iter).property_value();
		}

		std::vector<std::vector<double>> Legendre_val_con(nconditional_);

		//Legendre polynomial values
		std::vector<double> P(max_order + 1, 0.0);

		for( int i = 0; i < nconditional_; ++ i)
		{

			P[0] = 1;
			P[1] = conditional[i];
			//The above are the initialized values for the recursive computations
			
			//Compute the Legendre polynomials of the conditioning data up to the maximum order
			//We don't store the first two orders' polynomials since they are trivial
			for(int w = 1; w < max_order; ++ w)
			{
				P[w+1] = (P[1]*P[w]*(2*w+1) - w*P[w-1])/double(w+1);
				Legendre_val_con[i].push_back(P[w+1]);
			}

		}

		//////////////////////// 05 Feb 2018, Starting to compute the coefficients of Legendre polynomials of the conditional probability density function./////////////////////////

		//Initializtion of the vector of coefficients of Legendre polynomials 
		std::vector<double> coefs_pdf(max_order+1, 0);
		std::vector<double> coefs_array(max_order+1, 0);
		
		coefs_pdf_.resize(max_order + 1);


		int num_replicates = replicates_.size();
		//Going through all the replicates we've found
		for(int i = 0; i < num_replicates; ++ i)
		{
			//The center node of the template scanning over the training image
			coefs_array[0] = 1;
			coefs_array[1] = ti->selected_property()->get_value(replicates_[i][0]);

			for(int w = 2; w <= max_order; ++ w)
			{
				coefs_array[w] = Legendre_values[w-2]->get_value(replicates_[i][0]);
			}


			//For the rest of nodes in the replicate corresponding to the conditioning data.
			double coef_w;
			for( int n = 0; n < nconditional_; ++ n){

				//update the corresponding traning data and conditional data according to the position of
				//geometry template at the same time.
				double training_property = ti->selected_property()->get_value(replicates_[i][n+1]);
				double condition_property = conditional[n];

				//Update the coefficients
				coef_w = 0.5 + 1.5 * training_property * condition_property;

				for(int w = 2; w <= max_order; ++ w)
				{
					coef_w += (w + 0.5) 
							  * Legendre_values[w-2]->get_value(replicates_[i][n+1])
							  * Legendre_val_con[n][w-2];

				}

				//To multiply the coefficients of the new added conditional data
				for( int w = 0; w <= max_order; ++ w){
					
					coefs_array[w] *= coef_w;
				
				}

			}

			for(int w = 0; w <= max_order; ++ w)
			{
				coefs_pdf[w] +=coefs_array[w];
			}

		}

		for(int w = 1; w <= max_order; ++ w)
		{
			coefs_pdf[w] *= (w+0.5);

			coefs_pdf[w] /= coefs_pdf[0];

			coefs_pdf_[w] = coefs_pdf[w];
		}
	
		coefs_pdf[0] = 0.5;
		coefs_pdf_[0] = 0.5;

		return 0;
}


	//Write separate functions for different steps in the whole processing.
	//This functions serves to select a certain number of prototypes from a predefined set of prototypes.
	//The @prototype_moments stores the legendre moment values precomputed regarding the prototypes. 
	//The @prototype_legendre_values stores the legendre polynomial values precomputed regarding the prototypes. 


	int select_prototypes(const std::vector<std::vector<double> >& prototype_moments,
		const std::vector<std::vector<double> >& prototype_legendre_values,
		Gaussian_kernel_cdf & cdf,
		int num_sel
		)
	  {		
		std::vector<int> pdf_indices;
		std::vector<double> pdf_values;

		gap_ = 2.0/(double)(prototype_moments.size()-1);

		for(int i = 0; i < prototype_moments.size(); ++ i)
		{
			double pdf_value = 0.5;
			for(int w = 1; w <= cdf.order(); ++ w)
			{
				pdf_value += coefs_pdf_[w]*prototype_legendre_values[i][w];
			}
			if(pdf_value >= 0)
			{
				pdf_indices.push_back(i);
				pdf_values.push_back(pdf_value);
			}

		}

		//num_prototypes_ = prototype_moments.size();
		num_prototypes_ = pdf_values.size();

		std::vector<double> buffer_pdf_values(pdf_values);
		int num_selection = num_sel;
		//Select the n largest peaks as the center of the prototypes.		

		validates_.resize(num_selection);
		if(num_prototypes_ > num_selection)
		{
			//Find the nth-largest element.
			std::nth_element(buffer_pdf_values.begin(), buffer_pdf_values.begin() + num_selection-1, buffer_pdf_values.end(), std::greater<double>());
			int num_temp_sel = num_selection;
			int i = 0;
			int j = 0;
			while(j < num_selection)
			{
				if(pdf_values[i] >= buffer_pdf_values[num_selection-1])
					validates_[j++] = pdf_indices[i];
				++ i;
			}
			
			num_prototypes_ = num_selection;

		}
		else
		{
			std::copy(pdf_indices.begin(), pdf_indices.end(), validates_.begin());
			//If there happens to be prototypes less than the number to be selected
			num_prototypes_ = pdf_indices.size();
		}


		return 0;
}


	  //Initialize the memory of the QP matrices.
	  int initialize_QP()
	  {

		//The size of matrix equals to the number of replicates from the TI or data.
		//To reduce the size of the problem, we randomly pick some replicates from the TI currently.
		//Q.resize(num_replicates, num_replicates);
		Q.resize(num_prototypes_, num_prototypes_);

		//Initialize the q vector
		q.resize(0.0, num_prototypes_);

		//The constraint that sum of the weights to 1.
		CE.resize(num_prototypes_, 1);
		for(int i = 0; i < num_prototypes_; ++ i)
			CE[i][0] = 1.0;
		//It's one dimensional constraints
		ce0.resize(-1.0, 1);

		//The constraint that the weights be positive.
		CI.resize(0.0, num_prototypes_,num_prototypes_);
		for(int i = 0; i < num_prototypes_; ++ i)
			CI[i][i] = 1.0;
		ci0.resize(0.0, num_prototypes_);

		//The vector to store the solution to the quadratic programming.
		alpha.resize(num_prototypes_);

		return 0;

	  }

	  void duplicate_QP()
	  {
		  Q_dup = Q;
		  q_dup = q;
	  }
	  

	  //Fill in the entries of the QP matrices.
	  int build_QP(const std::vector<std::vector<double> >& prototype_moments, Gaussian_kernel_cdf & cdf, double lambda = 1.0e-7)
	  {
		//initilize the q vector!
		q.resize(0.0, num_prototypes_);

		double max_order = cdf.order();
		for(int i = 0; i < num_prototypes_; ++ i){
			for(int j = 0; j < num_prototypes_; ++ j){

				if(i > j)
					Q[i][j]=Q[j][i];
				else
				{	

					{
						double d = 0.0;
						for(int w = 0; w <= max_order; ++ w)
						{
							int ii = validates_[i];
							int jj = validates_[j];

							d += (w+0.5)*prototype_moments[ii][w]*prototype_moments[jj][w];
						}
						Q[i][j] = d;
					}
				}
			}

			double e;
			{
				int ii = validates_[i];

				e = coefs_pdf_[0] * prototype_moments[ii][0];

				e += coefs_pdf_[1] * prototype_moments[ii][1];
				//starting from order 2!!
				for(int w = 2; w <= max_order; ++ w)
				{
					e += coefs_pdf_[w] * prototype_moments[ii][w];
				}
	
			}
			q[i] -= e;
			Q[i][i] += lambda;

		}

		return 0;

	  }



	  //Solve the QP to get the weights for each prototype.
	  int solve_QP()
	  {
		  QuadProg::solve_quadprog(Q, q, CE, ce0, CI, ci0, alpha);
		  return 0;
	  }

	  //compute the objective function value of the QP
	  double compute_QP()
	  {
			//QuadProg::Vector<double> v = QuadProg::dot_prod(alpha, Q).extractRow(0);
			//double obj = QuadProg::dot_prod(v, alpha) * 0.5;
			//obj -= QuadProg::dot_prod(q, alpha);

		  //Q Matrix may change after solving the QP!!
			QuadProg::Vector<double> v = QuadProg::dot_prod(alpha, Q_dup).extractRow(0);
			//std::cout << alpha << std::endl;
			//std::cout << v << std::endl;
			//std::cout << Q_dup << endl;
			double obj = QuadProg::dot_prod(v, alpha) * 0.5;
			obj += QuadProg::dot_prod(q_dup, alpha);
			return obj;
	  }


	  //Get the estimated cdf from the QP
	  int estimate_cdf(Gaussian_kernel_cdf & cdf)
	  {

		//static int num_nodes = 0;

		  double denom = 0.0;

		  //Initialize the memory.
		  cdf.clear();
		for( int i = 0; i < num_prototypes_; ++ i)
		{
			double u = alpha[i];
			//double a = alpha[i];
			if(u < 1e-10 )
				continue;
			int ii = validates_[i];

			cdf.means_.push_back(-1+gap_*ii);//(ti->selected_property()->get_value(replicates[i][0]));
			cdf.coefs_.push_back(u);
			denom += u;

		}


		for( int i = 0; i < cdf.coefs_.size(); ++ i)
		{
			cdf.coefs_[i] /= denom;

		}
		cdf.balance();

		//std::cout<< "The simulation node is: NO. "<< ++ num_nodes <<std::endl;

		  return 0;
	  }


public:

	replicate_processor()
	{
	}


};



//Implementation of a Gausssian kernel denstiy estimator.

template <
	class Grid_ = Geostat_grid,
	class HardGrid_ = Point_set
>
class SLM_kde_estimator
{
public:

	SLM_kde_estimator(
		Grid_& ti,
		Grid_& hd,
		std::vector<GsTLGridProperty*>& Legendre_moments,
		std::vector<GsTLGridProperty*>& Legendre_values,
		std::vector<GsTLGridProperty*>& Legendre_values_hard,
		Avg_dummy_cdf& marginal,
		double lbound,
		double ubound,
		double angle_tol = 15,//in degrees
		double lag_tol = 5, // lag tolerance in grid dimension?
		double band_tol = 5, //bandwidth tolerance
		int num_replicates_ti = 1000,
		int num_replicates_hd = 100
		):  
	training_image_(ti),
		hard_data_(hd),
		Legendre_moments_(Legendre_moments),
		Legendre_values_(Legendre_values),
		Legendre_values_hard_(Legendre_values_hard),
		marginal_(marginal),
		lbound_(lbound),
		ubound_(ubound),
		angle_tol_(angle_tol),
		lag_tol_(lag_tol),
		band_tol_(band_tol),
		num_replicates_ti_(num_replicates_ti),
		num_replicates_hd_(num_replicates_hd)
	{
		initialize_path();
	}

  template <
            class Geovalue_,
            class Kernel_Cdf
           >
  inline int operator()(
			const Geovalue_& u,
			const Neighborhood& neighbors,
			Kernel_Cdf& ccdf
			)  ;


	void init_prototypes(Gaussian_kernel_cdf& cdf, int n, int num_sel, const char* file ="")
	{

		if( dynamic_cast<Strati_grid*>( &training_image_ ) ) {

			Strati_grid* sgrid = dynamic_cast<Strati_grid*>( &training_image_ );
			int num_replicates = sgrid->size();

			num_prototypes_ = n;
	
			prototype_moments_.resize(num_prototypes_);

			double delta = 2.0/(double)(num_prototypes_-1);
			prototype_legendre_values_.resize(num_prototypes_);
			//Initialize the derivatives 2018-10-12
			prototype_derivatives_.resize(num_prototypes_);

			for(int i = 0; i < num_prototypes_; ++ i)
			{
				Legendre_Gauss_Mom(cdf.order(), -1+delta*i, cdf.standard_deviation(), prototype_moments_[i], prototype_legendre_values_[i]);
			}
		}

		num_sel_protos_ = num_sel;
		if(strlen(file)){

			ofs_.open(file);

		}
	}


	void update_prototypes(Gaussian_kernel_cdf& cdf)
	{
		double delta = 2.0/(double)(num_prototypes_-1);
		for(int i = 0; i < num_prototypes_; ++ i)
		{
			//Legendre_Gauss_Mom(cdf.order(), -1+delta*i, cdf.standard_deviation(), prototype_moments_[i], prototype_legendre_values_[i], prototype_derivatives_[i]);
			Legendre_Gauss_Mom(cdf.order(), -1+delta*i, cdf.standard_deviation(), prototype_moments_[i], prototype_legendre_values_[i]);
		}
	}

	void close()
	{
		ofs_.close();
	}


private:

	std::vector<GsTLGridProperty*>& Legendre_moments_;
	std::vector<GsTLGridProperty*>& Legendre_values_;
	std::vector<GsTLGridProperty*>& Legendre_values_hard_;

	Avg_dummy_cdf& marginal_;
	std::vector<std::vector<double> > prototype_moments_;
	std::vector<std::vector<double> > prototype_legendre_values_;

	//To store the derivatives of the Legendre moments -- 12/10/2018
	std::vector<std::vector<double> > prototype_derivatives_;

	std::ofstream ofs_;



private:
	Grid_& training_image_; // The training image
	Grid_& hard_data_;

	double lag_tol_; //Lag tolerance
	double angle_tol_; //Angle tolerance
	double band_tol_; //Bandwidth tolerance

	int num_replicates_ti_; //The number of replicates from TI which may be randomly selected in order to reduce the calculating time
	int num_replicates_hd_; //The least number of hard data replicates needed, otherwise the replicates from TI are counted in

	int num_sel_protos_;

	int num_prototypes_;

	std::vector<int> random_ti_path_;

	//Only visit the nodes which has the hard data on the grid.
	std::vector<int> hard_ti_path_;

	double lbound_;
	double ubound_;

private:
	int initialize_path()
	{
		if( dynamic_cast<Strati_grid*>( &training_image_ ) ) {

			Strati_grid* sgrid = dynamic_cast<Strati_grid*>( &training_image_ );

			std::set<int> path;

			int nx = sgrid->nx();
			int ny = sgrid->ny();
			int nz = sgrid->nz();

			SGrid_cursor grid_cursor(nx, ny, nz);
			while( (int)path.size() < num_replicates_ti_){
				int i = rand() % nx;
				int j = rand() % ny;
				int k = rand() % nz;

				int node_id = grid_cursor.node_id(i, j, k);
				//Added on May 22 2018 to consider the training image inside a region or no-data-value.
				if(sgrid->selected_property()->is_informed(node_id) && sgrid->is_inside_selected_region(node_id))
					path.insert(node_id);
			}

			random_ti_path_.resize(path.size());

			std::copy(path.begin(), path.end(), random_ti_path_.begin());

		}

		if( dynamic_cast<Geostat_grid*>( &hard_data_ ) ) {

			Geostat_grid* hgrid = dynamic_cast<Strati_grid*>( &hard_data_ );

			GsTLGridProperty * prop = hgrid->selected_property();

			for(int i = 0; i < prop->size(); ++ i)
			{
				if(prop->is_informed(i))
					hard_ti_path_.push_back(i);
			}

			return 0;
		}

		return -1;
	}


};







//This is the main function to estimate the cpdf.

template <
	class Grid_,
	class HardGrid_
>
template <
          class Geovalue_,
          class Kernel_Cdf
         >
inline int 
SLM_kde_estimator<Grid_, HardGrid_>::operator()(
	     const Geovalue_& u,
	     const Neighborhood& neighbors,
	     Kernel_Cdf& cdf
	     ) 
{
	//Let's specialize the library to be GSTL_TNT_LIB for reusing the current code
	typedef matrix_lib_traits< GSTL_TNT_lib > MatrixLib;
	typedef Neighborhood::const_iterator const_iterator;


	//---parts of the code are just copied from the "kriging_weights.h"
	// If the neighborhood is empty, there is no kriging to be done.
	if( neighbors.is_empty() ) {
		gstl_warning( "Empty neighborhood." );
		return 2;
	} 


	//pass the work to TI & hard data scanning and processing
	replicate_processor kdp_ti;

	GsTLGridProperty* property = training_image_.selected_property();

	double mean = 0.0;
	for(const_iterator iter = neighbors.begin(); iter != neighbors.end(); ++ iter)
	{
		mean += (*iter).property_value();
	}
	mean /= (double) neighbors.size();
	marginal_.set_value(mean);

	int num_replicates = 0;

	double inner_rad = 3;
	//Using only the hard data as the training data.
	if(num_replicates_hd_ == 0)
	{
		num_replicates = kdp_ti.kernel_convolution( Geostat_grid::random_path_iterator( &hard_data_, property, 
						0,  
						hard_ti_path_.size(), 
						TabularMapIndex(&hard_ti_path_) ),
						Geostat_grid::random_path_iterator( &hard_data_, property, 
						hard_ti_path_.size(),  
						hard_ti_path_.size(), 
						TabularMapIndex(&hard_ti_path_) ),
						neighbors,
						cdf,
						Legendre_values_hard_);

		if (num_replicates<3)
			return -1;

	}
	//to use both the TI and the hard data
	else if (num_replicates_hd_ == 1) {
		//Scan the entire training image
		if (num_replicates_ti_ <= 0) {
			num_replicates = kdp_ti.kernel_convolution(Geostat_grid::random_path_iterator(&hard_data_, property,
				0,
				hard_ti_path_.size(),
				TabularMapIndex(&hard_ti_path_)),
				Geostat_grid::random_path_iterator(&hard_data_, property,
					hard_ti_path_.size(),
					hard_ti_path_.size(),
					TabularMapIndex(&hard_ti_path_)),
				training_image_.begin(),
				training_image_.end(),
				inner_rad,
				neighbors,
				cdf,
				Legendre_values_hard_,
				Legendre_values_);

		}
		else
		{
			//use selected number of replicates from the TI
			num_replicates = kdp_ti.kernel_convolution(Geostat_grid::random_path_iterator(&hard_data_, property,
				0,
				hard_ti_path_.size(),
				TabularMapIndex(&hard_ti_path_)),
				Geostat_grid::random_path_iterator(&hard_data_, property,
					hard_ti_path_.size(),
					hard_ti_path_.size(),
					TabularMapIndex(&hard_ti_path_)),
				Geostat_grid::random_path_iterator(&training_image_, property,
					0,
					random_ti_path_.size(),
					TabularMapIndex(&random_ti_path_)),
				Geostat_grid::random_path_iterator(&training_image_, property,
					random_ti_path_.size(),
					random_ti_path_.size(),
					TabularMapIndex(&random_ti_path_)),
				inner_rad,
				neighbors,
				cdf,
				Legendre_values_hard_,
				Legendre_values_);
		}

		if (num_replicates<3)
			return -1;
	}
	//Only using TI
	else {
		//using selected number of replicates from the TI
		if (num_replicates_ti_ > 0) {

			num_replicates = kdp_ti.search_replicates(Geostat_grid::random_path_iterator(&training_image_, property,
				0,
				random_ti_path_.size(),
				TabularMapIndex(&random_ti_path_)),
				Geostat_grid::random_path_iterator(&training_image_, property,
					random_ti_path_.size(),
					random_ti_path_.size(),
					TabularMapIndex(&random_ti_path_)),
				neighbors);

			if (num_replicates < 3)
				return -1;

			kdp_ti.fit_replicates(Geostat_grid::random_path_iterator(&training_image_, property,
				0,
				random_ti_path_.size(),
				TabularMapIndex(&random_ti_path_)),
				Geostat_grid::random_path_iterator(&training_image_, property,
					random_ti_path_.size(),
					random_ti_path_.size(),
					TabularMapIndex(&random_ti_path_)),
				neighbors,
				cdf,
				Legendre_values_);


		}
		else
		{
			//scan the entire TI to find the replicates
			num_replicates = kdp_ti.search_replicates(training_image_.begin(), training_image_.end(), neighbors);
			if (num_replicates < 3)
				return -1;

			kdp_ti.fit_replicates(training_image_.begin(), training_image_.end(), neighbors, cdf, Legendre_values_);

		}
	}

	//Build and solve the Quadratic programming problem
	{
		kdp_ti.select_prototypes(prototype_moments_,
									prototype_legendre_values_,
									cdf,
									num_sel_protos_);
		kdp_ti.initialize_QP();
		kdp_ti.build_QP(prototype_moments_, cdf);
		kdp_ti.solve_QP();
		kdp_ti.estimate_cdf(cdf);
	}


	if(num_replicates < 3)
		return -1;

	return 0;

}


//transform the original data into the interval [-1, 1] by proportion of the data distribution
template <class T>
class proportional_transform_legendre_domain
{
public:
	proportional_transform_legendre_domain(std::vector<double>& freqs, std::vector<double>& vals, double w)
		:freqs_(freqs),
		vals_(vals),
		width_(w)
	{
	}

	~proportional_transform_legendre_domain()
	{
	}

	void operator()(T& v)
	{
		double z = v;
		int t = 1;
		while(z > vals_[t] )
		{
			++t;
		}
		//v = (z-vals_[t-1])*width_/(vals_[t] - vals_[t-1]) + freqs_[t-1];
		v = (z-vals_[t-1])*(freqs_ [t]- freqs_ [t-1])/(vals_[t] - vals_[t-1]) + freqs_[t-1];//2020-05-04
		if (v > 1.0)
			v = 1.0;
		else if (v <= -1)
			v = -1.0;

	}


private:
	std::vector<double>& freqs_;
	std::vector<double>& vals_;
	double width_;

};



//Back transform the data form the interval [-1, 1] to the original data by proportion of the data distribution

class proportional_backfrom_legendre_domain
{
public:
	proportional_backfrom_legendre_domain(std::vector<double>& freqs, std::vector<double>& vals, double w)
		:freqs_(freqs),
		vals_(vals),
		width_(w)
	{
	}

	~proportional_backfrom_legendre_domain()
	{
	}
template <typename T>
	void operator()(T& v)
	{
		if(v >= 1.0)
		{
			v = vals_[vals_.size() - 1];
			return;
		}
		double f = v;
		int t = 1;
		while(f > freqs_[t])
		{
			++ t;
		}
		
		//v = (f - freqs_[t - 1])*(vals_[t] - vals_[t - 1]) / width_ + vals_[t - 1];
		v = (f - freqs_[t - 1])*(vals_[t] - vals_[t - 1]) / (freqs_[t] - freqs_[t - 1]) + vals_[t - 1];

	}


private:

	std::vector<double>& freqs_;
	std::vector<double>& vals_;
	double width_;
};

#endif