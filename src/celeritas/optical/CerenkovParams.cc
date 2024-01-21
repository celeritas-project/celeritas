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
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericGridData.hh"

#include "OpticalPropertyParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with optical property data.
 */
CerenkovParams::CerenkovParams(SPConstProperties properties)
{
    HostVal<CerenkovData> data;
    auto const& host_ref = properties->host_ref();
    DedupeCollectionBuilder reals(&data.reals);
    CollectionBuilder angle_integral(&data.angle_integral);
    for (auto mat_id :
         range(OpticalMaterialId(host_ref.refractive_index.size())))
    {
        GenericGridData ai_grid;
        auto const& ri_grid = host_ref.refractive_index[mat_id];
        if (!ri_grid)
        {
            // No refractive index data stored for this material
            angle_integral.push_back(ai_grid);
            continue;
        }

        // Calculate the Cerenkov angle integral
        auto const refractive_index = host_ref.reals[ri_grid.value];
        auto const energy = host_ref.reals[ri_grid.grid];
        std::vector<real_type> integral(energy.size());
        for (size_type i = 1; i < energy.size(); ++i)
        {
            integral[i] = integral[i - 1]
                          + real_type(0.5) * (energy[i] - energy[i - 1])
                                * (1 / ipow<2>(refractive_index[i - 1])
                                   + 1 / ipow<2>(refractive_index[i]));
        }
        ai_grid.grid = reals.insert_back(energy.begin(), energy.end());
        ai_grid.value = reals.insert_back(integral.begin(), integral.end());
        angle_integral.push_back(ai_grid);
    }
    data_ = CollectionMirror<CerenkovData>{std::move(data)};
    CELER_ENSURE(data_ || host_ref.refractive_index.empty());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
