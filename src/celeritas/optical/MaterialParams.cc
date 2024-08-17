//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MaterialParams.cc
//---------------------------------------------------------------------------//
#include "MaterialParams.hh"

#include <algorithm>
#include <utility>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/VectorUtils.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/GenericGridBuilder.hh"
#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/io/ImportData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<MaterialParams>
MaterialParams::from_import(ImportData const& data)
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
        input.properties.push_back(mat.second.properties);
    }
    return std::make_shared<MaterialParams>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with optical property data.
 */
MaterialParams::MaterialParams(Input const& inp)
{
    CELER_EXPECT(!inp.properties.empty());

    HostVal<MaterialParamsData> data;
    CollectionBuilder refractive_index{&data.refractive_index};
    GenericGridBuilder build_grid(&data.reals);
    for (auto const& mat : inp.properties)
    {
        // Store refractive index tabulated as a function of photon energy
        auto const& ri_vec = mat.refractive_index;
        if (ri_vec.x.empty())
        {
            // No refractive index data for this material
            refractive_index.push_back({});
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

        refractive_index.push_back(build_grid(ri_vec));
    }
    CELER_ASSERT(refractive_index.size() == inp.properties.size());

    data_ = CollectionMirror<MaterialParamsData>{std::move(data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
