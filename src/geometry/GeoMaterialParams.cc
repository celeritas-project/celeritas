//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoMaterialParams.cc
//---------------------------------------------------------------------------//
#include "GeoMaterialParams.hh"

#include <algorithm>
#include "base/CollectionBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from geometry and material params.
 */
GeoMaterialParams::GeoMaterialParams(Input input)
{
    CELER_EXPECT(input.geometry);
    CELER_EXPECT(input.materials);
    CELER_EXPECT(input.volume_to_mat.size() == input.geometry->num_volumes());
    CELER_EXPECT(std::all_of(input.volume_to_mat.begin(),
                             input.volume_to_mat.end(),
                             [input](MaterialId m) {
                                 return m.get()
                                        < input.materials->num_materials();
                             }));

    HostValue host_data;
    auto      materials = make_builder(&host_data.materials);
    materials.insert_back(input.volume_to_mat.begin(),
                          input.volume_to_mat.end());

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<GeoMaterialParamsData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
