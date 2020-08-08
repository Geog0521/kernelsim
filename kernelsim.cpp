
#include <algorithm>
#include <GsTLAppli/geostat/parameters_handler.h>
#include <GsTLAppli/geostat/utilities.h>
#include <GsTLAppli/utils/gstl_messages.h>
#include <GsTLAppli/utils/error_messages_handler.h>
#include <GsTLAppli/utils/string_manipulation.h>
#include <GsTLAppli/grid/grid_model/geostat_grid.h>
#include <GsTLAppli/grid/grid_model/combined_neighborhood.h>
#include <GsTLAppli/grid/grid_model/gval_iterator.h>
#include <GsTLAppli/grid/grid_model/cartesian_grid.h>
#include <GsTLAppli/grid/grid_model/point_set.h>
#include <GsTLAppli/appli/manager_repository.h>
#include <GsTLAppli/math/random_numbers.h>
#include <GsTLAppli/appli/utilities.h>

#include <GsTL/sampler/monte_carlo_sampler.h>
#include <GsTL/simulation/sequential_simulation.h>


//TESTING BEGIN
#include <GsTL/cdf/gaussian_cdf.h>
#include <GsTL/sampler/monte_carlo_sampler.h>
#include <GsTL/cdf_estimator/gaussian_cdf_Kestimator.h>
#include <GsTL/simulation/sequential_simulation.h>
#include <GsTL/univariate_stats/cdf_transform.h>
#include <GsTL/univariate_stats/build_cdf.h>
//TESTING END

#include <GsTLAppli/grid/grid_model/point_set_neighborhood.h>

#include <iterator>
#include <vector>
#include <algorithm>
#include <fstream>

#include <GsTLAppli/grid/grid_model/reduced_grid.h>


#include <boost/math/tools/roots.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/algorithm/minmax_element.hpp>
#include <boost/math/distributions/normal.hpp>


#include <qstring.h>
#include <qfile.h>

#include <qfiledialog.h>
#include <GsTLAppli/appli/project.h>
#include <GsTLAppli/utils/gstl_plugins.h>

#include "kernelsim.h"

kernelsim::kernelsim()
{
	simul_grid_ = 0;
	harddata_grid_ = 0;
	training_image_ = 0;
	training_property_ = 0;
	harddata_property_ = 0;
	assign_harddata_ = false;
	neighborhood_ = 0;
	multireal_property_ = 0;
	hard_data_ti_ = 0;

	temporary_harddata_property_ = 0;
	temporary_training_property_ = 0;

}

kernelsim::~kernelsim()
{
	clean();
}

//intialize the parameters related to the algorithm
bool kernelsim::initialize( const Parameters_handler* parameters,
		Error_messages_handler* errors )
{
	//-------------
	// Prepare the simulation grid and the property to be simulated.

	if (!get_simul_grid(parameters, errors))
	{
		return false;
	}

	//Set up the hard data

	if (!get_hard_data(parameters, errors))
	{
		return false;
	}

	//-------------
	// Set up the training image

	if (!get_training_image(parameters, errors))
	{
		return false;
	}

	//-------------
	// Set up the search neighborhood

	if (!set_up_neighborhood(parameters, errors))
	{
		return false;
	}

	//-------------
	// Set up the regions

	if (!set_up_regions(parameters, errors))
	{
		return false;
	}

	//-------------
	//// Set up the covariance
	//if(!set_up_covariance(parameters, errors))
	//{
	//	return false;
	//}

	//set up the bound values
	if(!set_bound_values(parameters, errors))
	{
		return false;
	}
	//test 0.5 2020-05-05
	build_frequency_table();
	
	//double c = 0;
	//std::vector<double> cutoffs;
	//while (c < 0.8) {
	//	c += 0.01;
	//	cutoffs.push_back(c);
	//}
	//cutoffs.push_back(1.0);
	//build_frequency_table(cutoffs);

	//You will have to tranform the properties to the interval [-1, 1] before you compute the moments.!!! 2018-02-13.

	//Use a transformation class to change the properties' values on hard grid and training image into [-1,1]
	if(temporary_harddata_property_)
		std::for_each(temporary_harddata_property_->begin(),temporary_harddata_property_->end(), transform_to_legendre_domain(zmin_, zmax_));
		//std::for_each(temporary_harddata_property_->begin(),temporary_harddata_property_->end(), proportional_transform_legendre_domain<PropertyValueProxy>(freq_table_, val_table_, 2*bin_width_));
	if(temporary_training_property_)
		std::for_each(temporary_training_property_->begin(),temporary_training_property_->end(), transform_to_legendre_domain(zmin_, zmax_));
		//std::for_each(temporary_training_property_->begin(),temporary_training_property_->end(), proportional_transform_legendre_domain<PropertyValueProxy>(freq_table_, val_table_, 2*bin_width_));


	///++++++++++++++++Here we store the hard data in a grid with the same size as the simulation grid+++++++++++++++
	//++++++++++++++++In addition, the data are preprocessed to computer the legendre polymials evaluated at these data values
	//++++++++++++++++For testing the hard-data-driven. 05/19/2019

	SmartPtr<Property_copier> tmp_property_copier_ = 
		Property_copier_factory::get_copier( harddata_grid_, hard_data_ti_);
	if( !tmp_property_copier_ ) {
		std::ostringstream message;
		message << "It is currently not possible to copy a property from a "
			<< harddata_grid_->classname() << " to a " 
			<< harddata_grid_->classname() ;
		errors->report( !tmp_property_copier_, "Transform_Hard_Data", message.str() );
		return false;
	}

	//clear the previous data stored in the hard data TI
	for(int i = 0; i < hard_data_ti_property_->size(); ++ i)
	{
		hard_data_ti_property_->set_not_informed(i);
	}

	tmp_property_copier_->copy(harddata_grid_, temporary_harddata_property_, hard_data_ti_, hard_data_ti_property_);
	//tmp_property_copier_->copy(harddata_grid_, harddata_property_, hard_data_ti_, hard_data_ti_property_);
	hard_data_ti_->select_property(hard_data_ti_property_->name());
	build_legendre_hard_ti(parameters, errors);

	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


	//currently just fixed for testing.
	//double sigma = 0.1;
	if(!build_kernel_moments(parameters, errors))
	{
		return false;
	}

	//-------------
	// Number of realizations and random number seed
	
	nb_of_realizations_ = 
		String_Op::to_number<int>( parameters->value( "Nb_Realizations.value" ) );
	
	seed_ = String_Op::to_number<int>( parameters->value( "Seed.value" ) );

	//-------------
	//Set the simple kriging type
	geostat_utils::KrigTagMap tags_map;
	geostat_utils::KrigDefaultsMap defaults;
	defaults[ geostat_utils::SK ] = "0.0";

	geostat_utils::Kriging_type ktype = geostat_utils::SK;
	geostat_utils::initialize( ktype, combiner_, Kconstraints_,
								tags_map,
								parameters, errors,
								simul_grid_, defaults );
	
	return true;
}

//main steps of the algorithm
int kernelsim::execute( GsTL_project* proj)
{




	// Set up a progress notifier	
	int total_steps = simul_grid_->size() * (nb_of_realizations_);
	int frequency = std::max( total_steps / 20, 1 );
	SmartPtr<Progress_notifier> progress_notifier = 
		utils::create_notifier( "Running kernelsim", 
		total_steps, frequency );




	int nb_multigrids = 3;
	int finest_grid_nb = 1;


	// work on the fine grid
	if( dynamic_cast<Strati_grid*>( simul_grid_ ) ) {
		Strati_grid* sgrid = dynamic_cast<Strati_grid*>( simul_grid_ );
		sgrid->set_level( finest_grid_nb );
	}

	//Set the training image to the finest resolution
	training_image_->set_level(finest_grid_nb);


	//To be changed later:
	//null_dummy_cdf marginal;
	Avg_dummy_cdf marginal;

	Gaussian_kernel_cdf ccdf;
	ccdf.order(max_order_);
	ccdf.learning_rate(learning_rate_);

	ccdf.standard_deviation(sigma_);
	//June 01, 2018 should'nt fxied the number of the prototypes now!
	//ccdf.num_prototypes(num_prototypes_);

  // set up the cdf-estimator
  typedef Gaussian_cdf_Kestimator< Covariance<Location>,
                                   Neighborhood,
                                   geostat_utils::KrigingConstraints
                                  >    Kriging_cdf_estimator;
  SLM_kde_estimator<> cdf_estimator( *(dynamic_cast<Geostat_grid*>(training_image_)),
							//*harddata_grid_,
							*hard_data_ti_,
							Legendre_moments_,
							Legendre_values_,
							Legendre_values_hard_ti_,
							marginal,
							zmin_,
							zmax_,
							angle_tol_,
							lag_tol_,
							band_tol_,
							num_ti_replicate_,
							num_hd_replicate_
						);

   cdf_estimator.init_prototypes(ccdf, num_prototypes_, num_sel_protos_);


	// Initialize the global random number generator
	Global_random_number_generator::instance()->seed( seed_ );

  // set up the sampler
  Random_number_generator gen;
  Monte_carlo_sampler_t< Random_number_generator > sampler( gen );
  

  bool from_scratch = true;
  // loop on all realizations
  for( int nreal = 0; nreal < nb_of_realizations_ ; nreal ++ ) {

    // compute the random path
    simul_grid_->init_random_path(from_scratch);
    from_scratch = false;

    // update the progress notifier
    progress_notifier->message() << "working on realization " 
                                 << nreal+1 << gstlIO::end;
    if( !progress_notifier->notify() ) return 1;


    // Create a new property to hold the realization and tell the simulation 
    // grid to use it as the current property 
    appli_message( "Creating new realization" );
    GsTLGridProperty* prop = multireal_property_->new_realization();
    simul_grid_->select_property( prop->name() );
    neighborhood_->select_property( prop->name() );

    // initialize the new realization with the hard data, if that was requested 
    if( property_copier_ ) {
      //Copy the property after transformation 2018-02-13
 
		property_copier_->copy( harddata_grid_, temporary_harddata_property_,
			simul_grid_, prop );
    }

    appli_message( "Doing simulation" );


	
    // do the simulation

	
	int status =
		sequential_simulation( simul_grid_->random_path_begin(),
			     simul_grid_->random_path_end(),
			     *(neighborhood_.raw_ptr()),
			     ccdf,
				 //cdf,
			     cdf_estimator,
			     marginal,
			     sampler, progress_notifier.raw_ptr()
			     );
    if( status == -1 ) {
      clean( prop );
      return 1;
    }

	std::cout<< "The number of bad points is: "<< status << std::endl;

	//At last, we should change the simulated values back to the original scale, because the previous results were drawn from [-1, 1].
	std::for_each(simul_grid_->selected_property()->begin(), simul_grid_->selected_property()->end(), back_from_legendre_domain(zmin_, zmax_));//(lowerbound_, upperbound_));
	//std::for_each(simul_grid_->selected_property()->begin(), simul_grid_->selected_property()->end(), proportional_backfrom_legendre_domain<PropertyValueProxy>(freq_table_, val_table_, 2*bin_width_));//(lowerbound_, upperbound_));

  }
  cdf_estimator.close();
  clean();

  return 0;
}

//create an instance of the algorithm
Named_interface* kernelsim::create_new_interface(std::string&)
{
	return new kernelsim;
}

//Get the simulation grid and specify the name of property to be simulated
bool kernelsim::get_simul_grid( const Parameters_handler* parameters,
		Error_messages_handler* errors )
{
	std::string simul_grid_name = parameters->value( "Grid_Name.value" );
	errors->report( simul_grid_name.empty(), 
		"Grid_Name", "No grid selected" );
	std::string property_name = parameters->value( "Property_Name.value" );
	errors->report( property_name.empty(), 
		"Property_Name", "No property name specified" );

	// Get the simulation grid from the grid manager  
	if( simul_grid_name.empty() ) return false;

	bool ok = geostat_utils::create( simul_grid_, simul_grid_name,
		"Grid_Name", errors );

	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	//Create a duplicate grid to hold the hard data in a Cartesian grid. 15/05/2019
	//ok = geostat_utils::create(hard_data_ti_ , simul_grid_name + ".hard",
	//	"Grid_Name", errors );


	//Need to create a new Cartesian grid to do this!!
	//if( !ok)
	{
		hard_ti_name_ = simul_grid_name + ".hard";
		std::string full_name( "/GridObject/Model/" + simul_grid_name + ".hard" );
		SmartPtr<Named_interface> ni = 
			Root::instance()->new_interface("cgrid://"+ simul_grid_name + ".hard", full_name);
  
		if( ni.raw_ptr() == 0 ) {
		errors->report( "Object " + full_name + " already exists. Use a different name." );
		//    appli_warning( "object " << full_name << "already exists" );
		return false;
		}
  
		Cartesian_grid* grid = dynamic_cast<Cartesian_grid*>( ni.raw_ptr() );
		Cartesian_grid* sgrid = dynamic_cast<Cartesian_grid*>( simul_grid_);
		grid->set_dimensions(
			sgrid->geometry()->dim(0), 
			sgrid->geometry()->dim(1), 
			sgrid->geometry()->dim(2),
			sgrid->cell_dimensions()[0], 
			sgrid->cell_dimensions()[1], 
			sgrid->cell_dimensions()[2]
		);

		grid->origin(sgrid->origin());
		//grid->set_rotation_z(sgrid->rotation_z());

		hard_data_ti_ = grid;
		ok = true;
	}

	hard_data_ti_property_ = hard_data_ti_->property(property_name);
	if(!hard_data_ti_property_)
		hard_data_ti_property_ = hard_data_ti_->add_property(property_name);

	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

	if( !ok ) return false;

	// create  a multi-realization property
	multireal_property_ = 
		simul_grid_->add_multi_realization_property( property_name );

	return true;
}

//retrieve the hard data grid
bool kernelsim::get_hard_data( const Parameters_handler* parameters,
		Error_messages_handler* errors )
{
	std::string harddata_grid_name = parameters->value( "Hard_Data.grid" );

	if( !harddata_grid_name.empty() ) {
		std::string hdata_prop_name = parameters->value( "Hard_Data.property" );
		errors->report( hdata_prop_name.empty(), 
			"Hard_Data", "No property name specified" );

		// Get the hard data grid from the grid manager
		bool ok = geostat_utils::create( harddata_grid_, harddata_grid_name, 
			"Hard_Data", errors );
		if( !ok ) return false;

		harddata_property_ = harddata_grid_->property( hdata_prop_name );
		if( !harddata_property_ ) {
			std::ostringstream error_stream;
			error_stream <<  harddata_grid_name 
				<<  " does not have a property called " 
				<< hdata_prop_name;
			errors->report( "Hard_Data", error_stream.str() );
			return false;
		}


		//Generate a temporary property that stores what is transformed to interval [-1,1] based on
		//the linear transformation of the original property, because the Legendre polynomials were
		//defined on D=[-1,1].
		std::string tmp_hdata_prop_name = hdata_prop_name + "temporary_";

		temporary_harddata_property_ = harddata_grid_->add_property(tmp_hdata_prop_name);

		if( !temporary_harddata_property_ ) {
			std::ostringstream error_stream;
			error_stream <<  harddata_grid_name 
				<<  " can not create a temporary property called " 
				<< tmp_hdata_prop_name;
			errors->report( "Hard_Data", error_stream.str() );
			return false;
		}
		SmartPtr<Property_copier> tmp_property_copier_ = 
			Property_copier_factory::get_copier( harddata_grid_, harddata_grid_ );

		if( !tmp_property_copier_ ) {
			std::ostringstream message;
			message << "It is currently not possible to copy a property from a "
				<< harddata_grid_->classname() << " to a " 
				<< harddata_grid_->classname() ;
			errors->report( !tmp_property_copier_, "Transform_Hard_Data", message.str() );
			return false;
		}

		tmp_property_copier_->copy(harddata_grid_, harddata_property_, harddata_grid_, temporary_harddata_property_);

	}



	// hard data assignment and transform is only needed if we have a valid
	// hard data grid and property.  We always assign the data if it belongs
	// the same grid

	assign_harddata_ = 
		String_Op::to_number<bool>( parameters->value( "Assign_Hard_Data.value" ) );
	if( harddata_grid_ == NULL ) assign_harddata_=false; 
	else if( harddata_grid_ == simul_grid_ ) assign_harddata_=true;

	if( assign_harddata_ ) {
		property_copier_ = 
			Property_copier_factory::get_copier( harddata_grid_, simul_grid_ );
		if( !property_copier_ ) {
			std::ostringstream message;
			message << "It is currently not possible to copy a property from a "
				<< harddata_grid_->classname() << " to a " 
				<< simul_grid_->classname() ;
			errors->report( !property_copier_, "Assign_Hard_Data", message.str() );
			return false;
		}
	} 

	return true;
}

//Get the instance of training image
bool kernelsim::get_training_image( const Parameters_handler* parameters,
		Error_messages_handler* error_mesgs )
{
	training_image_name_ = parameters->value( "PropertySelector_Training.grid" );
	error_mesgs->report( training_image_name_.empty(), 
		"PropertySelector_Training", "No training image selected" );

	training_property_name_ = parameters->value( "PropertySelector_Training.property" );
	error_mesgs->report( training_property_name_.empty(), 
		"PropertySelector_Training", "No training property selected" );

	// Get the training image from the grid manager
	// and select the training property
	if( !training_image_name_.empty() ) 
	{
		training_image_ = dynamic_cast<RGrid*>( 
			Root::instance()->interface( 
			gridModels_manager + "/" + training_image_name_).raw_ptr() );

		if( !training_image_ ) 
		{
			std::ostringstream error_stream;
			error_stream <<  training_image_name_ <<  " is not a valid training image";
			error_mesgs->report( "PropertySelector_Training", error_stream.str() );
			return false;
		}

		training_property_ = training_image_->property( training_property_name_ );
		if( !training_property_ ) {
			std::ostringstream error_stream;
			error_stream <<  training_image_name_ 
				<<  " does not have a property called " 
				<< training_property_name_;
			error_mesgs->report( "Training_Data", error_stream.str() );
			return false;
		}

		//Generate a temporary property that stores what is transformed to interval [-1,1] based on
		//the linear transformation of the original property, because the Legendre polynomials were
		//defined on D=[-1,1].
		std::string tmp_training_prop_name = training_property_name_ + "temporary_";

		temporary_training_property_ = training_image_->add_property(tmp_training_prop_name);

		if( !temporary_training_property_ ) {
			std::ostringstream error_stream;
			error_stream <<  training_image_name_ 
				<<  " can not create a temporary property called " 
				<< tmp_training_prop_name;
			error_mesgs->report( "Hard_Data", error_stream.str() );
			return false;
		}
		SmartPtr<Property_copier> tmp_property_copier_ = 
			Property_copier_factory::get_copier( training_image_, training_image_ );

		if( !tmp_property_copier_ ) {
			std::ostringstream message;
			message << "It is currently not possible to copy a property from a "
				<< training_image_->classname() << " to a " 
				<< training_image_->classname() ;
			error_mesgs->report( !tmp_property_copier_, "Transform_Training_Data", message.str() );
			return false;
		}

		tmp_property_copier_->copy(training_image_, training_property_, training_image_, temporary_training_property_);

		training_image_->select_property( tmp_training_prop_name );

		return true;
	}
	else 
		return false;
}


//Set the covariance model
bool kernelsim::set_up_covariance( const Parameters_handler* parameters,
		Error_messages_handler* errors )
{
  //-------------
  // Variogram (covariance) initialization 

  bool init_cov_ok = 
    geostat_utils::initialize_covariance( &covar_, "Variogram", 
                                          parameters, errors );

  return init_cov_ok;
}


//initialize the parameters for searching
bool kernelsim::set_up_neighborhood(const Parameters_handler* parameters,
	Error_messages_handler* errors)
{
	int max_neigh =
		String_Op::to_number<int>(parameters->value("Max_Conditioning_Data.value"));

	num_ti_replicate_ = String_Op::to_number<int>(parameters->value("num_ti_replicate.value"));
	num_hd_replicate_ = String_Op::to_number<int>(parameters->value("num_hd_replicate.value"));

	//Get the values for the tolerances to find a replicate in the hard data
	angle_tol_ = String_Op::to_number<double>(parameters->value("Angle_tol.value"));
	//Change to radian
	const static double pi = 3.14159265359;
	angle_tol_ = angle_tol_ * pi / 180.0;

	lag_tol_ = String_Op::to_number<double>(parameters->value("Lag_tol.value"));
	band_tol_ = String_Op::to_number<double>(parameters->value("Band_tol.value"));

	GsTLTriplet ranges;
	GsTLTriplet angles;
	bool extract_ok =
		geostat_utils::extract_ellipsoid_definition(ranges, angles,
			"Search_Ellipsoid.value",
			parameters, errors);
	if (!extract_ok) return false;

	// If the hard data are not "relocated" on the simulation grid,
	// use a "combined neighborhood", otherwise use a single 
	// neighborhood
	if (!harddata_grid_ || assign_harddata_) {

		neighborhood_ = SmartPtr<Neighborhood>(
			simul_grid_->neighborhood(ranges, angles));

	}
	else {
		Neighborhood* simul_neigh = simul_grid_->neighborhood(ranges, angles);

		simul_neigh->max_size(max_neigh);
		harddata_grid_->select_property(harddata_property_->name());

		Neighborhood* harddata_neigh;
		if (dynamic_cast<Point_set*>(harddata_grid_)) {
			harddata_neigh =
				harddata_grid_->neighborhood(ranges, angles, 0, true);
		}
		else {
			harddata_neigh =
				harddata_grid_->neighborhood(ranges, angles, 0);
		}


		harddata_neigh->max_size(max_neigh);
		//  harddata_neigh->select_property( harddata_property_->name() );

		neighborhood_ =
			SmartPtr<Neighborhood>(new Combined_neighborhood(harddata_neigh,
				simul_neigh));
		//     SmartPtr<Neighborhood>( new Combined_neighborhood_dedup( harddata_neigh,
	   //							                                           simul_neigh, &covar_, false) );
	}

	neighborhood_->max_size(max_neigh);

	// octant_2D_filter * filter = new octant_2D_filter(20, 6, 1, 5);

	double r = ranges[0];
	if (r < ranges[1])
		r = ranges[1];
	if (r < ranges[2])
		r = ranges[2];

	if (num_hd_replicate_  == 0 || num_hd_replicate_ == 1)
	{
		search_filter * filter = new search_filter(lag_tol_, band_tol_, angle_tol_, r + lag_tol_,  0.5*lag_tol_);
		neighborhood_->search_neighborhood_filter(filter);
	}


  //Do not include the center
  neighborhood_->includes_center(false);
  //For advanced parameters such as octant searching
  geostat_utils::set_advanced_search(neighborhood_, 
                      "AdvancedSearch", parameters, errors);

  return true;
}

//set the regions for simulation (not used here now)
bool kernelsim::set_up_regions( const Parameters_handler* parameters,
		Error_messages_handler* errors )
{
	std::string region_name = parameters->value( "Grid_Name.region" );
	if (!region_name.empty() && simul_grid_->region( region_name ) == NULL ) {
		errors->report("Grid_Name","Region "+region_name+" does not exist");
	}
	else grid_region_.set_temporary_region( region_name, simul_grid_);

	if(harddata_grid_ && !assign_harddata_ && harddata_grid_ != simul_grid_) {
		region_name = parameters->value( "Hard_Data.region" );
		if (!region_name.empty() && harddata_grid_->region( region_name ) == NULL ) {
			errors->report("Hard_Data","Region "+region_name+" does not exist");
		}
		else  hd_grid_region_.set_temporary_region( region_name,harddata_grid_ );
	}

	if(training_image_ != simul_grid_) {
		std::string region_name = parameters->value( "PropertySelector_Training.region" );
		if (!region_name.empty() && training_image_->region( region_name ) == NULL ) {
			errors->report("PropertySelector_Training","Region "+region_name+" does not exist");
		}
		else ti_grid_region_.set_temporary_region( region_name, training_image_);
		return true;
	}

	return true;
}


bool kernelsim::set_bound_values( const Parameters_handler* parameters, Error_messages_handler* errors )
{
	
	std::string orderstr = parameters->value("Maximum_order.value");
	if(orderstr.empty()){
		errors->report("Maximum_order", "Define an maximal order of Legenddre polynomials to approximate the cdf");
		return false;
	}

	max_order_ = String_Op::to_number<int>(orderstr);


	//I have to rewritten the minmax finding program, because the MS std::minmax_element or boost
	//is very, very slow to do this, probabily due to the safety bound check on the iterator with every visiting.
	//Although this can be avoided by define the _SECURE_SCL to 0, however this definition must be
	//consistent on every library (file) that use the STL, otherwise it will produce strange runtime errors.
	property_type zmax, zmin;
	
	if(harddata_property_){
		zmax = *(harddata_property_->begin());
		zmin = zmax;
		for(GsTLGridProperty::iterator iter = harddata_property_->begin(); iter != harddata_property_->end(); ++ iter){
			if(zmax < *iter)
				zmax = *iter;
			if(zmin > *iter)
				zmin = *iter;
		}
	}
	else{
		zmax = *(training_property_->begin());
		zmin = zmax;
	}
	
	for(GsTLGridProperty::iterator iter = training_property_->begin(); iter != training_property_->end(); ++ iter){
		if(zmax < *iter)
			zmax = *iter;
		if(zmin > *iter)
			zmin = *iter;
	}

	std::string strzmin = parameters->value("Min_value.value");
	std::string strzmax = parameters->value("Max_value.value");
	if(strzmin.empty()){
		errors->report("Min_value", "The input lower bound is required");
		return false;
	}

	if(strzmax.empty()){
		errors->report("Max_value", "The input upper bound is required");
		return false;
	}

	zmax_ = zmax;
	zmin_ = zmin;

	upperbound_ = zmax;
	lowerbound_ = zmin;

	zmax = String_Op::to_number<property_type>(strzmax);
	zmin = String_Op::to_number<property_type>(strzmin);

	if(zmax < zmax_){
		errors->report("Max_value", "The input upper bound is less than the existing data");
		return false;
	}
	else{
		zmax_ = zmax;
	}

	if(zmin > zmin_){
		errors->report("Min_value", "The input lower bound is greater than the existing data");
		return false;
	}
	else{
		zmin_ = zmin;
	}


	num_prototypes_ = String_Op::to_number<int>( parameters->value( "num_prototypes.value" ) );
	num_sel_protos_ = String_Op::to_number<int>( parameters->value( "num_sel_prototypes.value" ) );

	std::string str = parameters->value("optimize_check.value");
	if(str == "1")
	{
		b_optmize_width_ = true;
	}
	else 
		b_optmize_width_ = false;

	str = parameters->value("max_num_iteration.value");

	if(b_optmize_width_){
		if(str.empty()){
			errors->report("max_num_iteration", "Define the maximum iterations");
			return false;
		}

		num_max_iterations_ = String_Op::to_number<int>(str);
	}

	str = parameters->value("sigma_lower_bound.value");
	sigma_lb_ = String_Op::to_number<double>(str);

	str = parameters->value("sigma_upper_bound.value");
	sigma_ub_ = String_Op::to_number<double>(str);

	str = parameters->value("learning_rate.value");
	learning_rate_ = String_Op::to_number<double>(str);


	return true;

}


//free the memory usage
void kernelsim::clean( GsTLGridProperty* prop )
{
	if(prop)
		simul_grid_->remove_property( prop->name() );
	
	if(temporary_harddata_property_){
		harddata_grid_->remove_property(temporary_harddata_property_->name());
		temporary_harddata_property_ = 0;
	}
	if(temporary_training_property_){
		training_image_->remove_property(temporary_training_property_->name());
		temporary_training_property_ = 0;
	} 
	for(int i = 0; i < Legendre_moments_.size(); ++ i)
	{
		delete Legendre_moments_[i];
	}
	for(int i = 0; i < Legendre_values_.size(); ++ i)
	{
		delete Legendre_values_[i];
	}
	Legendre_moments_.clear();
	Legendre_values_.clear();

	if (hard_data_ti_)
	{
		//std::string str = hard_data_ti_->name();
		Root::instance()->delete_interface("/GridObject/Model/" + hard_ti_name_);// hard_data_ti_->name());
		hard_data_ti_ = 0;
	}

}


//Build the kernel moments and Legendre polynomial values for the training image
//These values will be used in the later steps to build the matrix for the Quadratic
//programming, since we are assuming now that all the replicates come from the training
//image.
bool kernelsim::build_kernel_moments( const Parameters_handler* parameters,
		Error_messages_handler* errors )
{

	std::string widthstr = parameters->value("kernel_width.value");
	if(widthstr.empty()){
		errors->report("kernel_width", "Define the kernel width");
		return false;
	}

	sigma_ = String_Op::to_number<double>(widthstr);

	//Initialize the moments up to W the same size as the training image.
	//We are working on the properties that are transformed to the interval [-1, 1].
	for(int w = 0; w <= max_order_; ++ w)
	{
		Legendre_moments_.push_back(new GsTLGridProperty(temporary_training_property_->size(), "moments"));
	}
	//We don't store the first two order of Legendre polynomials since they are trivial.
	for(int w = 0; w <= max_order_ - 2; ++ w)
	{
		Legendre_values_.push_back(new GsTLGridProperty(temporary_training_property_->size(), "values"));
	}


	//kermel moments
	std::vector<double> I(max_order_ + 1, 0.0);
	//intermediate variable to compute the kernel moments
	std::vector<double> T(max_order_ + 1, 0.0);
	//Legendre polynomial values
	std::vector<double> P(max_order_ + 1, 0.0);


	//Now let's go through the property array of the training image.
	for(int i = 0; i < temporary_training_property_->size(); ++ i)
	{
		//Added to skip non-data-value in the TI, May 22, 2018
		if(!temporary_training_property_->is_informed(i))
			continue;

		//we need to define the normal distribution with the mean as the node value and standard deviation as sigma at first.
		double mu = temporary_training_property_->get_value(i);
		boost::math::normal df(mu, sigma_);
		//I[0] = boost::math::cdf(df, 1) - boost::math::cdf(df, -1);
		//double c1 = sigma_*sigma_*(boost::math::pdf(df, -1) - boost::math::pdf(df, 1));
		//double c2 = sigma_*sigma_*(boost::math::pdf(df, 1) + boost::math::pdf(df, -1));
		double scaling = boost::math::cdf(df, 1) - boost::math::cdf(df, -1);
		I[0] = 1;
		//double c1 = sigma_*sigma_*(boost::math::pdf(df, -1) - boost::math::pdf(df, 1))/scaling;
		double c1 = sigma_*sigma_*(boost::math::pdf(df, 1) - boost::math::pdf(df, -1))/scaling;
		double c2 = sigma_*sigma_*(boost::math::pdf(df, 1) + boost::math::pdf(df, -1))/scaling;

		I[1] = mu * I[0] - c1;
		T[0] = I[1];
		P[0] = 1;
		P[1] = mu;
		//The above are the initialized values for the recursive computations

		Legendre_moments_[0]->set_value(I[0], i);
		Legendre_moments_[1]->set_value(I[1], i);


		for(int w = 1; w < max_order_; ++ w)
		{
			T[w] = mu * I[w] - (w%2?c2:c1);
			int k = w - 1;
			while (k >= 0)
			{
				T[w] += sigma_*sigma_* (2*k+1)*I[k];
				k -= 2;
			}


			I[w+1] = ((2*w+1)*T[w] - w*I[w-1])/double (w+1);


			P[w+1] = (mu*P[w]*(2*w+1) - w*P[w-1])/double(w+1);

			Legendre_moments_[w+1]->set_value(I[w+1], i);
			Legendre_values_[w-1]->set_value(P[w+1], i);
		}

	}

	return true;

}


//+++++++++++++precompute the legendre values for the hard data +++++++++++++++++++++++++++++ 
bool kernelsim::build_legendre_hard_ti( const Parameters_handler* parameters,
		Error_messages_handler* errors )
{

	//We don't store the first two order of Legendre polynomials since they are trivial.
	for(int w = 0; w <= max_order_ - 2; ++ w)
	{
		Legendre_values_hard_ti_.push_back(new GsTLGridProperty(hard_data_ti_property_->size(), "values"));
	}


	//kermel moments
	std::vector<double> I(max_order_ + 1, 0.0);
	//intermediate variable to compute the kernel moments
	std::vector<double> T(max_order_ + 1, 0.0);
	//Legendre polynomial values
	std::vector<double> P(max_order_ + 1, 0.0);


	//Now let's go through the property array of the training image.
	for(int i = 0; i < hard_data_ti_property_->size(); ++ i)
	{
		//Added to skip non-data-value in the TI, May 22, 2018
		if(!hard_data_ti_property_->is_informed(i))
			continue;

		//we need to define the normal distribution with the mean as the node value and standard deviation as sigma at first.
		double mu = hard_data_ti_property_->get_value(i);
		P[0] = 1;
		P[1] = mu;
		//The above are the initialized values for the recursive computations

		for(int w = 1; w < max_order_; ++ w)
		{
			P[w+1] = (mu*P[w]*(2*w+1) - w*P[w-1])/double(w+1);
			Legendre_values_hard_ti_[w-1]->set_value(P[w+1], i);
		}

	}

	return true;

}


//An copy and paster from the ::build_kernel_moments
void kernelsim::update_kernel_moments()
{
	//kermel moments
	std::vector<double> I(max_order_ + 1, 0.0);
	//intermediate variable to compute the kernel moments
	std::vector<double> T(max_order_ + 1, 0.0);
	//Legendre polynomial values
	std::vector<double> P(max_order_ + 1, 0.0);


	//Now let's go through the property array of the training image.
	for(int i = 0; i < temporary_training_property_->size(); ++ i)
	{
		//Added to skip non-data-value in the TI, May 22, 2018
		if(!temporary_training_property_->is_informed(i))
			continue;

		//we need to define the normal distribution with the mean as the node value and standard deviation as sigma at first.
		double mu = temporary_training_property_->get_value(i);
		boost::math::normal df(mu, sigma_);
		//I[0] = boost::math::cdf(df, 1) - boost::math::cdf(df, -1);
		//double c1 = sigma_*sigma_*(boost::math::pdf(df, -1) - boost::math::pdf(df, 1));
		//double c2 = sigma_*sigma_*(boost::math::pdf(df, 1) + boost::math::pdf(df, -1));
		double scaling = boost::math::cdf(df, 1) - boost::math::cdf(df, -1);
		I[0] = 1;
		//double c1 = sigma_*sigma_*(boost::math::pdf(df, -1) - boost::math::pdf(df, 1))/scaling;
		double c1 = sigma_*sigma_*(boost::math::pdf(df, 1) - boost::math::pdf(df, -1))/scaling;
		double c2 = sigma_*sigma_*(boost::math::pdf(df, 1) + boost::math::pdf(df, -1))/scaling;




		I[1] = mu * I[0] - c1;
		T[0] = I[1];
		P[0] = 1;
		P[1] = mu;
		
		//The above are the initialized values for the recursive computations




		Legendre_moments_[0]->set_value(I[0], i);
		Legendre_moments_[1]->set_value(I[1], i);





		for(int w = 1; w < max_order_; ++ w)
		{
	

			T[w] = mu * I[w] - (w%2?c2:c1);
			int k = w - 1;
			while (k >= 0)
			{
				T[w] += sigma_*sigma_* (2*k+1)*I[k];
				k -= 2;
			}


			I[w+1] = ((2*w+1)*T[w] - w*I[w-1])/double (w+1);


			P[w+1] = (mu*P[w]*(2*w+1) - w*P[w-1])/double(w+1);


			Legendre_moments_[w+1]->set_value(I[w+1], i);
			Legendre_values_[w-1]->set_value(P[w+1], i);
		}

	}
}

void kernelsim::compute_derivative()
{
	

}


//Optimze the kernel width by stochastic gradient descent. June 15 2018
int kernelsim::optimize_kernel_width()
{


	return 0;
}


void kernelsim::build_frequency_table(double bin_width)
{
	bin_width_ = bin_width;
	if(!temporary_harddata_property_)
		return;
	//Copy the data vaules to a new vector
	std::vector<double> values(temporary_harddata_property_->size());

	for(int i = 0; i < values.size(); ++ i)
	{
		values[i] = temporary_harddata_property_->get_value(i);
	}

	//sort the values
	//CAN'T UNDERSTAND WHY SORT FUNCTION DOES NOT WORK ON THIS MACHINE!
	//std::sort(values.begin(), values.end());

	//
	int num_bins = int (1.0/bin_width);
	freq_table_.resize(num_bins+1);
	val_table_.resize(num_bins+1);

	freq_table_[0] = -1;
	val_table_[0] = zmin_;

	for(int i = 1; i < num_bins; ++ i)
	{
		freq_table_[i] = freq_table_[i-1] + bin_width*2;
		int n = int (bin_width * i * values.size());

		std::nth_element(values.begin(), values.begin() + n - 1, values.end());
		val_table_[i] = values[n];
	}

	freq_table_[num_bins] = 1;
	//val_table_[num_bins] = 12;
	val_table_[num_bins] = zmax_;

}

void kernelsim::build_frequency_table(std::vector<double>& cutoffs)
{
	if (!temporary_harddata_property_)
		return;
	//Copy the data vaules to a new vector
	std::vector<double> values(temporary_harddata_property_->size());

	for (int i = 0; i < values.size(); ++i)
	{
		values[i] = temporary_harddata_property_->get_value(i);
	}

	//
	int num_bins = cutoffs.size();
	freq_table_.resize(num_bins);
	val_table_.resize(num_bins);

	freq_table_[0] = -1;
	val_table_[0] = zmin_;

	for (int i = 1; i < num_bins-1; ++i)
	{
		freq_table_[i] = freq_table_[i - 1] + (cutoffs[i]-cutoffs[i-1])* 2;
		int n = int(cutoffs[i] * values.size());

		std::nth_element(values.begin(), values.begin() + n - 1, values.end());
		val_table_[i] = values[n];
	}

	freq_table_[num_bins-1] = 1;
	val_table_[num_bins - 1] = zmax_;
	//val_table_[num_bins - 1] = 12;

}



//inversion of the probability
double Base_kernel_cdf::inverse(double p) const
{
	double result = 0.0;

	//boost::uintmax_t max_iter = 1000;

	static const double lim=1.0e-12;
	static const double INFINITY=GsTL::INFINITY;

	std::pair<double, double> root = boost::math::tools::bisect(
		//boost::BOOST_BIND(&Truncated_Legendre_cdf::prob,this,_1),
		Base_kernel_cdf::root_find_helper (const_cast<Base_kernel_cdf*> (this), p),
		-1.0,
		1.0,
		//-INFINITY,
		//INFINITY,
		basic_toleration()//,
		//max_iter
		);

 
	result = (root.first + root.second)/2;

	return result;
}


#include <boost/math/distributions/normal.hpp>


//Compute the value of probability Prob(Z <= z)
double Gaussian_kernel_cdf::prob( double z ) const
{
	double p = 0.0;


	//double s = std::accumulate(coefs_.begin(), coefs_.end(), 0.0);
	//double s = std::sqrt(variance_);
	

	for (int i = 0; i < means_.size(); ++ i)
	{
		//boost::math::normal c(means_[0], s);
		//double x = boost::math::cdf(c, z);
		//p += coefs_[i] * x;

		Gaussian_cdf cdf(means_[i], variance_);//0.2);
		//x = cdf.prob(z);
		
		//if(coefs_[i]==0.0)
		//	continue;
		//p += coefs_[i] * cdf.prob(z);
		//To shift accoring to the definition of a truncated normal distribution.
		p += coefs_[i] * (cdf.prob(z) - shift_[i])/scaling_[i];
		//p += coefs_[i] * cdf.prob(z);


	}

	//p -= overall_shift_;
	//p /= overall_scaling_;

	//if(p<0)
	//	p = 0;
	//else if(p>1)
	//	p = 1.0;

	//assert( p>=0 && p<=1);

	return p;
}

void Gaussian_kernel_cdf::balance()
{
	if(balanced_)
		return;

	shift_.resize(means_.size());
	scaling_.resize(means_.size());

	overall_shift_ = 0;
	overall_scaling_ = 0;

	for (int i = 0; i < means_.size(); ++ i)
	{
		//boost::math::normal c(means_[0], s);
		//double x = boost::math::cdf(c, z);
		//p += coefs_[i] * x;
		Gaussian_cdf cdf(means_[i], variance_);

		//To shift accoring to the definition of a truncated normal distribution.
		shift_[i] =  cdf.prob(-1.0);
		double s = cdf.prob(1.0) - shift_[i];
		scaling_[i] = s;

		overall_shift_ += coefs_[i] * shift_[i];
		overall_scaling_ += coefs_[i] * s;

	}


	balanced_ = true;
}



//Some helpful functions related to the algorithm of reproducing kernel generated by Legendre polynomials.


//This function compute the moment of a univariate Legendre polynomial with normal distribution on the interval [-1,1].
//Store the moments up to w in the vector
int Legendre_Gauss_Mom(int max_order, double mu, double sigma, std::vector<double>& moments, std::vector<double>& P, std::vector<double>& prototype_derivatives)
{
	//We only consider the moments of positive orders
	assert (max_order >= 1);

	//Moments from order 0 to order w
	//moments.resize(max_order+1, 0.0);
	moments.assign(max_order+1, 0.0);

	//Derivatives corresponding to the moments on sigma 2018-10-12
	//prototype_derivatives.resize(max_order+1, 0.0);
	prototype_derivatives.assign(max_order+1, 0.0);

	//intermediate variable to compute the kernel moments
	std::vector<double> T(max_order + 1, 0.0);
	//Legendre polynomial values
	//P.resize(max_order + 1, 0.0);
	P.assign(max_order + 1, 0.0);


	//The vector to store intermediate values to compute derivatives 12/10/2018
	std::vector<double> DI(max_order + 1, 0.0);
	std::vector<double> DT(max_order + 1, 0.0);

		boost::math::normal df(mu, sigma);
		//June 04, 2018. IMPORTANT, we actually need a truncated normal distribution on [-1, 1] here to be the prototypes.
		//OTHERWISE, the integral of the pdf is not 1 and will intend to generate extreme high values in the simulation.
		//moments[0] = boost::math::cdf(df, 1) - boost::math::cdf(df, -1);
		//double c1 = sigma*sigma*(boost::math::pdf(df, -1) - boost::math::pdf(df, 1));
		//double c2 = sigma*sigma*(boost::math::pdf(df, 1) + boost::math::pdf(df, -1));

		double scaling = boost::math::cdf(df, 1) - boost::math::cdf(df, -1);
		moments[0] = 1.0; 
		double c1 = sigma*sigma*(boost::math::pdf(df, 1) - boost::math::pdf(df, -1))/scaling;
		double c2 = sigma*sigma*(boost::math::pdf(df, 1) + boost::math::pdf(df, -1))/scaling;
		moments[1] = mu * moments[0] - c1;
		T[0] = moments[1];
		P[0] = 1;
		P[1] = mu;
		//The above are the initialized values for the recursive computations


		//The derivatives of initialized values --Added on 12/10/2018
		double C1 = ((1-mu)*(1-mu)/sigma + sigma)/scaling * boost::math::pdf(df, 1) - ((1+mu)*(1+mu)/sigma + sigma)/scaling  * boost::math::pdf(df, -1)
			+ ((1-mu)*boost::math::pdf(df, 1) + (1+mu)*boost::math::pdf(df, -1))*c1/sigma/scaling;

		double C2 = ((1-mu)*(1-mu)/sigma+ sigma)/scaling  * boost::math::pdf(df, 1) + ((1+mu)*(1+mu)/sigma + sigma)/scaling * boost::math::pdf(df, -1)
			+ ((1-mu)*boost::math::pdf(df, 1) + (1+mu)*boost::math::pdf(df, -1))*c2/sigma/scaling;

		//
		DI[0] = 0;
		DI[1] = -C1;
		DT[0] = DI[1];
		//The above are intialized to compute the derivatives recursively.

		//It's a waste of memory to store the constant value DI[0], but just for the temporary convenience.
		prototype_derivatives[0] = DI[0];
		prototype_derivatives[1] = DI[1];

		for(int w = 1; w < max_order; ++ w)
		{
			DT[w] = mu * DI[w] - (w%2?C2:C1);

			T[w] = mu * moments[w] - (w%2?c2:c1);
			int k = w - 1;
			while (k >= 0)
			{
				DT[w] += (sigma*sigma*DI[k] + 2*sigma*moments[k])*(2*k+1);

				T[w] += sigma*sigma* (2*k+1)*moments[k];
				k -= 2;
			}

						
			DI[w+1] = ((2*w+1)*DT[w] - w*DI[w-1])/double (w+1);
			P[w+1] = (mu*P[w]*(2*w+1) - w*P[w-1])/double(w+1);

			prototype_derivatives[w+1] = DI[w+1];
			moments[w+1] = ((2*w+1)*T[w] - w*moments[w-1])/double (w+1);
		}






	////Let's calculate E[P_0] at first, not that P_0(x) = 1.
	//boost::math::normal c(m, 1.0); 

	//double EP0 =  boost::math::cdf(c, 1-m) - boost::math::cdf(c, -1-m);
	////Now let's compute E[P_1] and note thata P_1(x) = x.

	////Check the derivation on the blue notebook.(YLQ)
	//double EP1 = m * EP0 + boost::math::pdf(c, -1-m) - boost::math::pdf(c, 1-m);

	//moments[0] = EP0;
	//if (w == 0){		
	//	return 0;
	//}
	//moments[1] = EP1;
	//if (w==1){
	//	
	//	return 0;
	//}

	//double c1 = boost::math::pdf(c, 1);
	//double c2 = boost::math::pdf(c, -1);

	////Now calculate the moments based on the recursive relations.
	////Check the derivation from the blue notebook.
	//double dP0 = EP0;
	//double dP1 = EP1;
	//for(int n = 2; n <= w; ++ n){

	//	double realn = (double) n;
	//	//note the property of Pn(-1) = (-1)^n and Pn(1) = 1
	//	double cn = ((n%2)? c2:-c2) - c1;		


	//	//Compute integral of Gau(x)dP_(n-1)
	//	//for( int i = n - 1; i > 0; i -= 2){
	//		//dP += (2*i + 1) * moments[i];
	//	//}
	//	
	//	moments[n] =  ( (2 * realn - 1) *(cn + m * moments[n-1] + ((n%2)?dP0:dP1)) - 
	//					(realn - 1) * moments[n-2] ) / realn;
	//	if(n%2){
	//		dP0 += moments[n];
	//	}
	//	else{
	//		dP1 += moments[n];
	//	}
	//}

	return 0;
}


//This function compute the moment of a univariate Legendre polynomial with normal distribution on the interval [-1,1].
//Store the moments up to w in the vector
int Legendre_Gauss_Mom(int max_order, double mu, double sigma, std::vector<double>& moments, std::vector<double>& P)
{
	//We only consider the moments of positive orders
	assert (max_order >= 1);

	//Moments from order 0 to order w
	//moments.resize(max_order+1, 0.0);
	moments.assign(max_order+1, 0.0);


	//intermediate variable to compute the kernel moments
	std::vector<double> T(max_order + 1, 0.0);
	//Legendre polynomial values
	//P.resize(max_order + 1, 0.0);
	P.assign(max_order + 1, 0.0);


		boost::math::normal df(mu, sigma);
		//June 04, 2018. IMPORTANT, we actually need a truncated normal distribution on [-1, 1] here to be the prototypes.
		//OTHERWISE, the integral of the pdf is not 1 and will intend to generate extreme high values in the simulation.
		//moments[0] = boost::math::cdf(df, 1) - boost::math::cdf(df, -1);
		//double c1 = sigma*sigma*(boost::math::pdf(df, -1) - boost::math::pdf(df, 1));
		//double c2 = sigma*sigma*(boost::math::pdf(df, 1) + boost::math::pdf(df, -1));

		double scaling = boost::math::cdf(df, 1) - boost::math::cdf(df, -1);
		moments[0] = 1.0; 
		double c1 = sigma*sigma*(boost::math::pdf(df, 1) - boost::math::pdf(df, -1))/scaling;
		double c2 = sigma*sigma*(boost::math::pdf(df, 1) + boost::math::pdf(df, -1))/scaling;
		moments[1] = mu * moments[0] - c1;
		T[0] = moments[1];
		P[0] = 1;
		P[1] = mu;
		//The above are the initialized values for the recursive computations



		for(int w = 1; w < max_order; ++ w)
		{

			T[w] = mu * moments[w] - (w%2?c2:c1);
			int k = w - 1;
			while (k >= 0)
			{
				T[w] += sigma*sigma* (2*k+1)*moments[k];
				k -= 2;
			}

			P[w+1] = (mu*P[w]*(2*w+1) - w*P[w-1])/double(w+1);

			moments[w+1] = ((2*w+1)*T[w] - w*moments[w-1])/double (w+1);
		}


	return 0;
}

GEOSTAT_PLUGIN(kernelsim)