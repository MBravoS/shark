//
// ICRAR - International Centre for Radio Astronomy Research
// (c) UWA - The University of Western Australia, 2018
// Copyright by UWA (in the framework of the ICRAR)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//

/**
 * @file
 */

#ifndef INCLUDE_ENVIRONMENT_H_
#define INCLUDE_ENVIRONMENT_H_



#include <memory>
#include <utility>

#include "components.h"
#include "dark_matter_halos.h"
#include "options.h"

namespace shark {

class EnvironmentParameters{

public:
	explicit EnvironmentParameters(const Options &options);

	bool gradual_stripping = false;
	bool stripping = true;

};

class Environment{

public:
	explicit Environment(const EnvironmentParameters &parameters);

	void process_satellite_subhalo_environment (Subhalo &satellite_subhalo, Subhalo &central_subhalo);

private:

	EnvironmentParameters parameters;
};

using EnvironmentPtr = std::shared_ptr<Environment>;

template <typename ...Ts>
EnvironmentPtr make_environment(Ts&&...ts)
{
	return std::make_shared<Environment>(std::forward<Ts>(ts)...);
}


}//end namespace shark


#endif /* INCLUDE_ENVIRONMENT_H_ */
