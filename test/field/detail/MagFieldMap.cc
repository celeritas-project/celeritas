//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MagFieldMap.cc
//---------------------------------------------------------------------------//
#include "MagFieldMap.hh"

#include "base/Assert.hh"
#include "base/CollectionBuilder.hh"

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

    HostValue host_group;
    this->build_data(load_map, &host_group);

    // Move to mirrored data, copying to device
    group_ = CollectionMirror<detail::FieldMapData>{std::move(host_group)};
    CELER_ENSURE(this->group_);
}

//---------------------------------------------------------------------------//
/*!
 * Convert an input map to a MagFieldMap and store to FieldMapData.
 */
void MagFieldMap::build_data(ReadMap load_map, HostValue* group)
{
    CELER_EXPECT(load_map);
    detail::FieldMapData result = load_map();

    group->params = result.params;

    auto fieldmap = make_builder(&group->fieldmap);

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
