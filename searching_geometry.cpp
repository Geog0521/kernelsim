#include "searching_geometry.h"
#include <iomanip>
std::ostream& operator<< (std::ostream &out, const octant_2D_table &tbl)
{
	int dim = (int)(tbl.radius_ / tbl.cell_size_ + 0.5);

	int i = 0;
	int j = 0;
	for (int y = dim; y >= -dim; --y) {

		j = y + dim;
		for (int x = -dim; x <= dim; ++x)
		{
			i = x + dim;

			if (tbl.table_[j*(2 * dim + 1) + i] == -1)
				out << std::setw(5) << ' ';
			else
				out << std::setw(5) << tbl.table_[j*(2 * dim + 1) + i];

		}
		out << std::endl;
	}

	return out;
}


bool octant_2D_filter::is_admissible( const Geovalue& neigh, const Geovalue& center)
{
	typedef Geovalue::location_type location_type;
	typedef location_type::difference_type Euclidean_vector;

	//Get the directional lag for each canditate point inside the neighborhood
	Euclidean_vector diff = neigh.location() - center.location();

	if (diff.x()*diff.x() + diff.y()*diff.y() <= radius_exclu_*radius_exclu_)
	{
		return false;
	}


	//Get the sector number based on the lag.
	int sect = oct_table_.sector(diff.x(), diff.y());
	//If not sector number is found, it means that the point is not eligible to be a conditioning data
	if (sect == -1)
		return false;
	
	//The candidate is found in a valid zone for the first time
	//We need to store the sequence number of the conditioning data which increases as the number of
	//conditioning data that already exist. The sequence number should be propagated also to the adjacent
	//zones of the current zone so that the successive points would not overlap with the current zone.
	//Since the list of candidate points are sorted increasingly according to their distances to the center node,
	//here we don't need to maintain the list of points fallen in the same zone for comparing the closeness to the
	//center node.
	if(sectors_[sect] == -1){
		//Record the sequence number
		


		//Propagate the sequence no is easy in 2D situation, which are the sequence_no +/- 1
		//the current zone is 24*n, then the adjacent would be 24n + 23 other than sect-1, otherwise, the adjacent zone is (sect - 1).

		int adj_sect_dec = sect % 24 == 0 ? 23 + sect : sect - 1;
		//The similar happens to the end of the ring for increment 1 of the sequence No. 
		int adj_sect_inc = (sect+1) % 24 == 0 ? sect - 23 : sect + 1;
		sectors_[adj_sect_dec] = sequence_no_;
		sectors_[sect] = sequence_no_; // informed_sectors_.size();//1
		sectors_[adj_sect_inc] = sequence_no_;

		//Store the adjacent zones that already get informed for later restoration
		informed_sectors_.push_back(adj_sect_dec);
		informed_sectors_.push_back(sect);
		informed_sectors_.push_back(adj_sect_inc);

		++ sequence_no_;

		return true;
	}
	else{
		return false;
	}

	//return diff*diff > lag_tol_*lag_tol_;
}


void octant_2D_filter::clear()
{
	for(int i = 0; i < informed_sectors_.size(); ++ i)
	{
		sectors_[informed_sectors_[i]] = -1;
	}

	informed_sectors_.clear();
	sequence_no_ = 0;
}

//Match the node from the data event to the replicate from the sample data
int octant_2D_filter::match(double x, double y, double z)
{
	int sect = oct_table_.sector(x, y);

	if(sect != -1)
		return sectors_[sect];

	return -1;
}


double octant_2D_filter::search_radius()
{
	return oct_table_.radius();
}


bool search_filter::is_admissible(const Geovalue& neigh, const Geovalue& center)
{


	typedef Geovalue::location_type location_type;
	typedef location_type::difference_type Euclidean_vector;

	//Get the directional lag for each canditate point inside the neighborhood
	Euclidean_vector v = neigh.location() - center.location();

	//double d = 0;
	//for (int i = 0; i < GsTLGridVector::dimension; ++i)
	//{
	//	//The candidate lag
	//	d += v[i] * v[i];
	//}
	////Exclude points inside a certain range
	////if (d <= radius_exclu_*radius_exclu_)
	////{
	////	return false;
	////}


	//for (int ind = 0; ind < geoms_.size(); ++ind)
	//{
	//	//int ind = *iter;
	//	double l = 0;

	//	double a = 0;
	//	for (int i = 0; i < GsTLGridVector::dimension; ++i)
	//	{
	//		//The reference lag
	//		a += geoms_[ind][i] * geoms_[ind][i];
	//		//
	//		l += v[i] * geoms_[ind][i];

	//	}

	//	double la = std::sqrt(a);
	//	double l_proj = l / la;
	//	//out of the range of the lag tolerance
	//	if (l_proj < la - lag_tol_/2 || l_proj > la + lag_tol_/2)
	//	{
	//		continue;
	//	}
	//	double l_perp = std::sqrt(d - l_proj * l_proj);

	//	double db = std::sqrt(d) * std::sin(angle_tol_);

	//	double band = db <= l_perp ? db : l_perp;

	//	//out of the range of the bandwidth
	//	if (l_perp > band/2)
	//	{
	//		continue;
	//	}

	//	return false;
	//}

	//update the admissible list of neighbor points
	geoms_.push_back(v);

	return true;
}

int search_filter::match(int nid, int sz, GsTLGridVector& v, std::list<matching_info>& matchings)
{
	int matched = 0;
	if (sz <= 0)
		sz = geoms_.size();
	for (int ind = 0; ind < sz; ++ind)
	{
		//int ind = *iter;
		double l = 0;
		double d = 0;
		double a = 0;
		for (int i = 0; i < GsTLGridVector::dimension; ++i)
		{
			//The reference lag
			a += geoms_[ind][i] * geoms_[ind][i];
			//
			l += v[i] * geoms_[ind][i];
			//The candidate lag
			d += v[i] * v[i];

		}

		double la = std::sqrt(a);
		//keep the inner part for the TI training
		//if (la < lag_tol_)
		//	continue;

		double l_proj = l / la;
		//out of the range of the lag tolerance
		if (l_proj < la - lag_tol_ || l_proj > la + lag_tol_)
		{
			continue;
		}
		double l_perp = std::sqrt(d - l_proj * l_proj);

		double db = std::sqrt(d) * std::sin(angle_tol_);

		double band = db <= l_perp ? db : l_perp;

		//out of the range of the bandwidth
		if (l_perp > band)
		{
			continue;
		}

		double dist = 0;
		for (int i = 0; i < GsTLGridVector::dimension; ++i) {
			//Get the distance between the candidate and the matched position
			dist += (v[i] - geoms_[ind][i])*(v[i] - geoms_[ind][i]);
		}

		matching_info info;
		info.visiting_id = matchings.size();
		info.node_id = nid;
		//Changed to ind + 1 to include the center node. 2020-03-22
		info.matched_id = ind + 1;
		info.matched_distance = dist;

		//sorting the list of matchings increasingly by the distance
		std::list<matching_info>::iterator iter_mat = matchings.begin();
		while (iter_mat != matchings.end())
		{
			d = (*iter_mat).matched_distance;
			if (dist <= d)
				break;
			++iter_mat;
		}
		matchings.insert(iter_mat, info);
		++matched;

		//sorting by the distance is done

	}


	return matched;
}


void search_filter::clear()
{
	geoms_.clear();
}