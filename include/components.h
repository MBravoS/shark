//
// Base classes that make up a SHArk solving system.
//
// ICRAR - International Centre for Radio Astronomy Research
// (c) UWA - The University of Western Australia, 2017
// Copyright by UWA (in the framework of the ICRAR)
// All rights reserved
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston,
// MA 02111-1307  USA
//

#ifndef SHARK_COMPONENTS_H_
#define SHARK_COMPONENTS_H_

#include <algorithm>
#include <map>
#include <memory>
#include <ostream>
#include <vector>

#include "mixins.h"

namespace shark {

// Forward-defines
class Galaxy;
class Subhalo;
class Halo;
class MergerTree;

typedef std::shared_ptr<Galaxy> GalaxyPtr;
typedef std::shared_ptr<Subhalo> SubhaloPtr;
typedef std::shared_ptr<Halo> HaloPtr;
typedef std::shared_ptr<MergerTree> MergerTreePtr;


/**
 * The common base for all baryon component types.
 */
class BaryonBase {

public:

	/**
	 * Initialize values in zero.
	 */
	BaryonBase():
		mass(0),
		mass_metals(0)
	{
		// no-op
	}
	/**
	 * Mass content of the baryon component
	 */
	float mass;

	/**
	 * Metallicity of the baryon component
	 */
	float mass_metals;
};

/**
 * A common baryon component.
 * Note that black holes are not baryon components as they use their own class.
 */
class Baryon : public BaryonBase {
public:
	/**
	 * Initialize values in zero.
	 */
	Baryon():
		rscale(0),
		sAM(0)
	{
		// no-op
	}
	/**
	 * A scale radius
	 */
	float rscale;

	/**
	 * Specific angular momentum
	 */
	float sAM;
};

/**
 * Black hole baryon component.
 *
 * Because of the no hair theorem, black holes are only allowed to have a mass,
 * an accretion rate and a spin. This implies that to extend black holes from
 * the basic galaxy_component class, one needs to include at most an accretion
 * rate and a spin.
 */
class BlackHole : public BaryonBase {

public:

	/**
	 * Initialize values in zero.
	 */
	BlackHole():
		macc(0)
	{
		// no-op
	}
	/**
	 * macc: accretion rate onto the black hole.
	 */
	float macc;
};

/**
 * A basic galaxy.
 *
 * Galaxies have at least a bulge, a disk and a black hole. The most basic
 * galaxy allowed in SHArk contains a disk and a bulge, with either component
 * having gas and stars. A Galaxy also requires to have a SMBH.
 */
class Galaxy : public Identifiable<int> {
public:

	/**
	 * Initialize values in zero.
	 */
	Galaxy():
		tmerge(0),
		galaxy_type(),
		sfr_disk(0),
		sfr_bulge(0)
	{
		//no-op
	}

	/**
	 * An enumeration of types of galaxies
	 */
	enum galaxy_type_t {
		CENTRAL = 0,
		TYPE1,
		TYPE2,
		FLYBY
	};

	/**
	 * The type of galaxy
	 */
	galaxy_type_t galaxy_type;

	Baryon bulge_stars;
	Baryon bulge_gas;
	Baryon disk_stars;
	Baryon disk_gas;
	BlackHole smbh;

	//save average star formation rates.
	float sfr_disk;
	float sfr_bulge;

	/**
	 * dynamical friction timescale, which is defined only is galaxy is satellite.
	 */
	float tmerge;

	/**
	 * Define functions to calculate total mass and metals of various components.
	 */

	double disk_mass(){
		return disk_gas.mass + disk_stars.mass;
	}

	double disk_mass_metals(){
		return disk_gas.mass_metals + disk_stars.mass_metals;
	}

	double bulge_mass(){
		return bulge_gas.mass + bulge_stars.mass;
	}

	double bulge_mass_metals(){
		return bulge_gas.mass_metals + bulge_stars.mass_metals;
	}

	double baryon_mass(){
		return disk_gas.mass + disk_stars.mass + bulge_gas.mass + bulge_stars.mass;
	}

	double stellar_mass(){
		return disk_stars.mass + bulge_stars.mass;
	}

	double stellar_mass_metals(){
		return disk_stars.mass_metals + bulge_stars.mass_metals;
	}

	double gas_mass(){
		return disk_gas.mass + bulge_gas.mass;
	}

	double gas_mass_metals(){
		return disk_gas.mass_metals + bulge_gas.mass_metals;
	}


	double composite_size(){


		double rdisk = 0;
		double rbulge = 0;

		if(disk_mass() > 0){
			rdisk = (disk_stars.mass * disk_stars.rscale + disk_gas.mass * disk_gas.rscale) / disk_mass();
		}
		if(bulge_mass() > 0){
			rbulge = (bulge_stars.mass * bulge_stars.rscale + bulge_gas.mass * bulge_gas.rscale) / bulge_mass();
		}

		double rcomp = 0.0;

		// Define rcomp only if galaxy has mass.
		if(baryon_mass() > 0){
			rcomp = (disk_mass() * rdisk + bulge_mass() * rbulge) / baryon_mass();
		}

		return rcomp;
	}
};

/** This class extends the galaxy to include spatial information.*/
class SpatialGalaxy : public Galaxy, public Spatial<float> {
};

/** Extend galaxy to be satellite by including a merging timescale. */
class SatelliteGalaxy : public Galaxy {
public:
	float tmerge;
};

/** This class extends the satellite galaxy to include spatial information.*/
class SpatialSatelliteGalaxy : public SatelliteGalaxy, public Spatial<float> {
};


/**
 * This structure keeps track of the properties of the halo gas, which are necessary to implement a more sophisticated cooling model.
 */
struct CoolingSubhaloTracking {
	/**
	 * Initialize values in zero.
	 */
	CoolingSubhaloTracking():
		deltat(),
		temp(),
		mass(),
		tcooling()
	{
		//no=op
	};
	std::vector<double> deltat;
	std::vector<double> temp;
	std::vector<double> mass;
	std::vector<double> tcooling;
};


/** This class defines what a subhalo is. In an abstract sense, a subhalo is the quantum units of how dark matter clusters. Subhalos can
 * coexist with other subhalos in the same halo. A subhalo con also host 0, 1 or more galaxies depending on how we allow galaxies to populate
 * subhalos.
 */
class Subhalo : public Identifiable<long>, public Spatial<float> {

public:

	/**
	 * Initialize values in zero.
	 */
	Subhalo():
		haloID(0),
		has_descendant(false),
		main_progenitor(false),
		descendant_id(0),
		descendant_halo_id(0),
		snapshot(-1),
		descendant_snapshot(-1),
		last_snapshot_identified(-1),
		subhalo_type(),
		Vvir(0),
		Mvir(0),
		L{0, 0, 0},
		Vcirc(0),
		concentration(),
		accreted_mass(0),
		descendant(0),
		galaxies(),
		ascendants(),
		host_halo()
	{
		//no-op
	}

	/**
	 * An enumeration of types of subhalos
	 */
	enum subhalo_type_t {
		CENTRAL = 0,
		SATELLITE,
		FLYBY
	};

	/**
	 * The snapshot at which this subhalo is found
	 */
	int snapshot;

	/**
	 * Whether this subhalo has a descendant or not
	 */
	bool has_descendant;

	/**
	 * Boolean property indicating if subhalo is a main progenitor of its descendant.
	 */
	bool main_progenitor;


	/**
	 * The ID of the descendant of this subhalo.
	 * Valid only if has_descendant is \code{true}
	 */
	id_t descendant_id;

	/**
	 * The ID of the Halo containing the descendant of this subhalo
	 */
	id_t descendant_halo_id;

	/**
	 * The snapshot at which the descendant of this subhalo can be found
	 */
	int descendant_snapshot;

    /**
     * Integer that shows if this subhalo will disappear from the tree in the next snapshot.
     * last_snapshot_identified = 1 if disappears in the next snapshot, =0 otherwise.
     */

    int last_snapshot_identified;

	/**
	 * A pointer to the descendant of this subhalo.
	 * If this pointer is set then descendant_id and descendant_subhalo are
	 * meaningless.
	 */
	SubhaloPtr descendant;

	/**
	 * The list of galaxies in this subhalo.
	 */
	std::vector<GalaxyPtr> galaxies;

	/**
	 * The subhalo type
	 */
	subhalo_type_t subhalo_type;

	/**
	 * Which Halo does this Subhalo belong to
	 */
	id_t haloID;

	/** TODO: Properly document these */
	float Vvir;
	float Mvir;
	xyz<float> L;
	float Vcirc;
	float concentration;

	/**
	 * This component saves que information of the virial temperature, total halo gas and cooling time history.
	 */
	CoolingSubhaloTracking cooling_subhalo_tracking;


	/**
	 * Hot gas component of the halo and outside the galaxies that is
	 * allowed to cool down and/or fall onto the galaxy.
	 */
	Baryon hot_halo_gas;

	/**
	 * Cold gas component of the halo and outside the galaxies that has
	 * cooled down.
	 */
	Baryon cold_halo_gas;

	/**
	 * Hot gas component of the halo and outside galaxies that tracks
	 * the ejected outflowing gas from the galaxy and that is not
	 * available for cooling yet.
	 */
	Baryon ejected_galaxy_gas;


	/**
	 * A list of pointers to the ascendants of this subhalo, sorted by mass in
	 * descending order
	 */
	std::vector<SubhaloPtr> ascendants;

	/**
	 * The accreted mass onto the subhalo. This information comes from the merger tree.
	 */
	float accreted_mass;

	/**
	 * The halo that holds this subhalo.
	 */
	HaloPtr host_halo;

	// Sort ascendant subhalos by Mvir
	std::vector<SubhaloPtr> ordered_ascendants(){

		if(ascendants.size()==0){
			return std::vector<SubhaloPtr>();
		}
		else if(ascendants.size()>1){
			std::sort(ascendants.begin(), ascendants.end(), [](const SubhaloPtr &lhs, const SubhaloPtr &rhs) {
			return lhs->Mvir > rhs->Mvir;
			});
		}

		return ascendants;

	}

	/// Returns a pointer to the central galaxy. If no central galaxy is found
	/// in this Subhalo, then an empty pointer is returned.
	GalaxyPtr central_galaxy(){
		for (auto galaxy: galaxies){
			if(galaxy->galaxy_type == Galaxy::CENTRAL){
				return galaxy;
			}
		}
		return GalaxyPtr();
	}

	/// Copies the galaxies from this Subhalo into `target`
	void copy_galaxies_to(SubhaloPtr &target) {
		target->galaxies.insert(target->galaxies.end(), galaxies.begin(), galaxies.end());
	}

	/// Transfers (i.e., moves) the galaxies from this Subhalo into `target`
	void transfer_galaxies_to(SubhaloPtr &target) {
		copy_galaxies_to(target);
		galaxies.clear();
	}

};

template <typename T>
std::basic_ostream<T> &operator<<(std::basic_ostream<T> &stream, const Subhalo &subhalo)
{
	stream << "<Subhalo " << subhalo.id;
	if (subhalo.host_halo) {
		stream << " @ " << subhalo.host_halo;
	}
	stream << ">";
	return stream;
}

template <typename T>
std::basic_ostream<T> &operator<<(std::basic_ostream<T> &stream, const SubhaloPtr &subhalo)
{
	stream << *subhalo;
	return stream;
}

/**
 * Class to extend the properties a subhalo can have, by allowing it to have more baryon components than the basic subhalo.
 */
class SuperSubhalo : public Subhalo {
public:
	float rscale;
	float sAM;
	Baryon ejected_galaxy_gas; /*Gas that has been ejected by the galaxy but lives inside the halo.*/
	Baryon ejected_halo_gas; /*Gas that has been ejected outside the halo.*/
	Baryon halo_stars; /*Stars that live in the halo and outside the galaxy.*/
};


/**
 * A halo.
 *
 * Halos are the largest gravitationally bound structures in the universe. They
 * must contain at least one subhalo inside.
 */
class Halo : public Identifiable<long>, public Spatial<float> {

public:

	Halo(Halo::id_t halo_id, int snapshot) :
		central_subhalo(),
		satellite_subhalos(),
		mass_fraction_subhalos(-1),
		Vvir(0),
		Mvir(0),
		concentration(0),
		snapshot(snapshot)
	{
		// no-op
		id = halo_id;
	}

	/**
	 * The central subhalo
	 */
	SubhaloPtr central_subhalo;

	/**
	 * The subhalos contained in this halo
	 */
	std::vector<SubhaloPtr> satellite_subhalos;

	std::vector<SubhaloPtr> all_subhalos() {

		std::vector<SubhaloPtr> all;

		if (central_subhalo) {
			all.push_back(central_subhalo);
		}
		all.insert(all.end(), satellite_subhalos.begin(), satellite_subhalos.end());

		// If there are more than one subhalo, then return them ordered by mass in decreasing order.
		if(all.size() > 1){
			std::sort(all.begin(), all.end(), [](const SubhaloPtr &lhs, const SubhaloPtr &rhs) {
				return lhs->Mvir > rhs->Mvir;
			});
		}

		return all;
	}

	/**
	 * The mass contained in the subhalos.
	 * This quantity should be =1 for classic SAMs, but with Rodrigo Canas work
	 * on VELOCIraptor, this quantity could be less than 1.
	 */
	float mass_fraction_subhalos;

	/** TODO: document these */
	float Vvir;
	float Mvir;
	float concentration;

	/**
	 * The snapshot at which this halo is found
	 */
	int snapshot;

	HaloPtr descendant;
	std::vector<HaloPtr> ascendants;

	/**
	 * The merger tree that holds this halo.
	 */
	MergerTreePtr merger_tree;

	void add_subhalo(const SubhaloPtr &subhalo) {

		// Assign subhalo to proper member
		if (subhalo->subhalo_type == Subhalo::CENTRAL) {
			central_subhalo = subhalo;
		}
		else {
			satellite_subhalos.push_back(subhalo);
		}

		// Add subhalo mass to halo
		Mvir += subhalo->Mvir;
	}

};

template <typename T>
std::basic_ostream<T> &operator<<(std::basic_ostream<T> &stream, const Halo &halo)
{
	stream << "<Halo " << halo.id;
	if (halo.merger_tree) {
		stream << " @ " << halo.merger_tree;
	}
	stream << ">";
	return stream;
}

template <typename T>
std::basic_ostream<T> &operator<<(std::basic_ostream<T> &stream, const HaloPtr &halo)
{
	stream << *halo;
	return stream;
}

/**
 * A merger tree.
 *
 * A merger tree contains halos, which are indexed by snapshot,
 * and an ID to identify it.
 */
class MergerTree : public Identifiable<long> {
public:

	/**
	 * All halos contained in this merger tree, indexed by snapshot number
	 */
	std::map<int, std::vector<HaloPtr>> halos;

	void add_halo(const HaloPtr &halo) {
		halos[halo->snapshot].push_back(halo);
	}
};

template <typename T>
std::basic_ostream<T> &operator<<(std::basic_ostream<T> &stream, const MergerTree &merger_tree)
{
	stream << "<MergerTree " << merger_tree.id << ">";
	return stream;
}

template <typename T>
std::basic_ostream<T> &operator<<(std::basic_ostream<T> &stream, const MergerTreePtr &merger_tree)
{
	stream << *merger_tree;
	return stream;
}

}  // namespace shark

#endif // SHARK_COMPONENTS_H_
