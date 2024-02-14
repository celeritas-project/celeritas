//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalPropertyParams.cc
//---------------------------------------------------------------------------//
#include "OpticalPropertyParams.hh"

#include <algorithm>
#include <utility>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/grid/VectorUtils.hh"
#include "celeritas/io/ImportData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<OpticalPropertyParams>
OpticalPropertyParams::from_import(ImportData const& data)
{
    CELER_EXPECT(!data.optical.empty());

    if (!std::any_of(
            data.optical.begin(), data.optical.end(), [](auto const& iter) {
                return static_cast<bool>(iter.second.properties);
            }))
    {
        // No optical property data present
        return nullptr;
    }

    Input input;
    for (auto const& mat : data.optical)
    {
        input.data.push_back(mat.second.properties);
    }
    return std::make_shared<OpticalPropertyParams>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with optical property data.
 */
OpticalPropertyParams::OpticalPropertyParams(Input const& inp)
{
    HostVal<OpticalPropertyData> data;
    DedupeCollectionBuilder reals(&data.reals);
    CollectionBuilder refractive_index(&data.refractive_index);
    for (auto const& mat : inp.data)
    {
        // Store refractive index tabulated as a function of photon energy
        auto const& ri_vec = mat.refractive_index;
        GenericGridData grid;
        if (ri_vec.x.empty())
        {
            // No refractive index data for this material
            refractive_index.push_back(grid);
            continue;
        }

        // In a dispersive medium the index of refraction is an increasing
        // function of photon energy
        CELER_VALIDATE(is_monotonic_increasing(make_span(ri_vec.x)),
                       << "refractive index energy grid values are not "
                          "monotonically increasing");
        CELER_VALIDATE(is_monotonic_increasing(make_span(ri_vec.y)),
                       << "refractive index values are not monotonically "
                          "increasing");

        grid.grid = reals.insert_back(ri_vec.x.begin(), ri_vec.x.end());
        grid.value = reals.insert_back(ri_vec.y.begin(), ri_vec.y.end());
        CELER_ASSERT(grid);
        refractive_index.push_back(grid);
    }
    data_ = CollectionMirror<OpticalPropertyData>{std::move(data)};
    CELER_ENSURE(data_ || inp.data.empty());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
