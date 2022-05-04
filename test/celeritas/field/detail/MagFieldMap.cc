//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file field/detail/MagFieldMap.cc
//---------------------------------------------------------------------------//
#include "MagFieldMap.hh"

#include "corecel/Assert.hh"
#include "corecel/data/CollectionBuilder.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a user-defined field map.
 */
MagFieldMap::MagFieldMap(ReadMap load_map)
{
    CELER_ENSURE(load_map);

    HostValue host_data;
    this->build_data(load_map, &host_data);

    // Move to mirrored data, copying to device
    mirror_ = CollectionMirror<detail::FieldMapData>{std::move(host_data)};
    CELER_ENSURE(this->mirror_);
}

//---------------------------------------------------------------------------//
/*!
 * Convert an input map to a MagFieldMap and store to FieldMapData.
 */
void MagFieldMap::build_data(const ReadMap& load_map, HostValue* host_data)
{
    CELER_EXPECT(load_map);
    detail::FieldMapInput result = load_map();

    host_data->params = result.params;

    auto fieldmap = make_builder(&host_data->fieldmap);

    size_type ngrid = result.params.num_grid_z * result.params.num_grid_r;
    fieldmap.reserve(ngrid);

    for (auto i : range(ngrid))
    {
        fieldmap.push_back(result.data[i]);
    }
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
