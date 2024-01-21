//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalPropertyParams.cc
//---------------------------------------------------------------------------//
#include "OpticalPropertyParams.hh"

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
    CELER_EXPECT(!data.materials.empty());

    Input input;
    for (auto const& mat : data.materials)
    {
        if (!mat.optical_properties)
        {
            continue;
        }

        OpticalMaterial optical_mat;
        auto const& vec_map = mat.optical_properties.vectors;
        if (auto iter = vec_map.find(ImportOpticalVector::refractive_index);
            iter != vec_map.end())
        {
            optical_mat.refractive_index = iter->second;
        }
        input.materials.push_back(optical_mat);
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
    for (auto const& mat : inp.materials)
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
        CELER_ASSERT(is_monotonic_increasing(make_span(ri_vec.x)));
        CELER_ASSERT(is_monotonic_increasing(make_span(ri_vec.y)));

        grid.grid = reals.insert_back(ri_vec.x.begin(), ri_vec.x.end());
        grid.value = reals.insert_back(ri_vec.y.begin(), ri_vec.y.end());
        CELER_ASSERT(grid);
        refractive_index.push_back(grid);
    }
    data_ = CollectionMirror<OpticalPropertyData>{std::move(data)};
    CELER_ENSURE(data_ || inp.materials.empty());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
