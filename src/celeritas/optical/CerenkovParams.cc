//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CerenkovParams.cc
//---------------------------------------------------------------------------//
#include "CerenkovParams.hh"

#include <utility>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericCalculator.hh"
#include "celeritas/grid/GenericGridData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with optical property data.
 */
CerenkovParams::CerenkovParams(OpticalPropertyHostRef const& properties)
{
    HostVal<CerenkovData> data;
    for (auto mat_id :
         range(OpticalMaterialId(properties.refractive_index.size())))
    {
        GenericGridData grid_data;
        auto const& refractive_index = properties.refractive_index[mat_id];
        if (!refractive_index)
        {
            // No refractive index data stored for this material
            make_builder(&data.angle_integral).push_back(grid_data);
            continue;
        }

        // Calculate the Cerenkov angle integral
        GenericCalculator calc_rindex(refractive_index, properties.reals);
        auto const energy = properties.reals[refractive_index.grid];
        std::vector<real_type> integral(energy.size());
        for (size_type i = 1; i < energy.size(); ++i)
        {
            integral[i] = integral[i - 1]
                          + real_type(0.5) * (energy[i] - energy[i - 1])
                                * (1 / ipow<2>(calc_rindex[i - 1])
                                   + 1 / ipow<2>(calc_rindex[i]));
        }
        auto reals = make_builder(&data.reals);
        grid_data.grid = reals.insert_back(energy.begin(), energy.end());
        grid_data.value = reals.insert_back(integral.begin(), integral.end());
        make_builder(&data.angle_integral).push_back(grid_data);
    }
    data_ = CollectionMirror<CerenkovData>{std::move(data)};
    CELER_ENSURE(data_ || properties.refractive_index.empty());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
