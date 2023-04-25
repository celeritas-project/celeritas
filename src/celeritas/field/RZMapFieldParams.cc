//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldParams.cc
//---------------------------------------------------------------------------//
#include "RZMapFieldParams.hh"

#include "corecel/Assert.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Units.hh"

#include "RZMapFieldInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a user-defined field map.
 */
RZMapFieldParams::RZMapFieldParams(RZMapFieldInput const& inp)
{
    CELER_VALIDATE(inp.num_grid_z >= 2,
                   << "invalid field parameter (num_grid_z=" << inp.num_grid_z
                   << ")");
    CELER_VALIDATE(inp.num_grid_r >= 2,
                   << "invalid field parameter (num_grid_r=" << inp.num_grid_r
                   << ")");
    CELER_VALIDATE(inp.delta_grid > 0,
                   << "invalid field parameter (delta_grid=" << inp.delta_grid
                   << ")");
    CELER_VALIDATE(
        inp.field_z.size() == inp.num_grid_z * inp.num_grid_r,
        << "invalid field length (field_z size=" << inp.field_z.size()
        << "): should be " << inp.num_grid_z * inp.num_grid_r);
    CELER_VALIDATE(
        inp.field_r.size() == inp.field_z.size(),
        << "invalid field length (field_r size=" << inp.field_r.size()
        << "): should be " << inp.field_z.size());

    auto host_data = [&inp] {
        HostVal<RZMapFieldParamsData> host;

        host.params.num_grid_r = inp.num_grid_r;
        host.params.num_grid_z = inp.num_grid_z;
        host.params.delta_grid = inp.delta_grid;
        host.params.offset_z = inp.offset_z;

        auto fieldmap = make_builder(&host.fieldmap);
        fieldmap.reserve(inp.field_z.size());
        for (auto i : range(inp.field_z.size()))
        {
            // Save field vector, converting from Tesla to native units
            FieldMapElement el;
            el.value_z = inp.field_z[i] * units::tesla;
            el.value_r = inp.field_r[i] * units::tesla;
            fieldmap.push_back(el);
        }
        return host;
    }();

    // Move to mirrored data, copying to device
    mirror_ = CollectionMirror<RZMapFieldParamsData>{std::move(host_data)};
    CELER_ENSURE(this->mirror_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
