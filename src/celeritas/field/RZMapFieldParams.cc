//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldParams.cc
//---------------------------------------------------------------------------//
#include "RZMapFieldParams.hh"

#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/UniformGridData.hh"
#include "celeritas/Units.hh"

#include "RZMapFieldData.hh"
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
    CELER_VALIDATE(inp.min_r >= 0,
                   << "invalid field parameter (min_r=" << inp.min_r << ")");
    CELER_VALIDATE(inp.max_r > inp.min_r,
                   << "invalid field parameter (max_r=" << inp.max_r
                   << " <= min_r= " << inp.min_r << ")");
    CELER_VALIDATE(inp.max_z > inp.min_z,
                   << "invalid field parameter (max_z=" << inp.max_z
                   << " <= min_z= " << inp.min_z << ")");

    CELER_VALIDATE(
        inp.field_z.size() == inp.num_grid_z * inp.num_grid_r,
        << "invalid field length (field_z size=" << inp.field_z.size()
        << "): should be " << inp.num_grid_z * inp.num_grid_r);
    CELER_VALIDATE(
        inp.field_r.size() == inp.field_z.size(),
        << "invalid field length (field_r size=" << inp.field_r.size()
        << "): should be " << inp.field_z.size());

    // Throw a runtime error if any driver options are invalid
    validate_input(inp.driver_options);

    auto host_data = [&inp] {
        HostVal<RZMapFieldParamsData> host;

        host.grids.data_r = UniformGridData::from_bounds(
            {inp.min_r, inp.max_r}, inp.num_grid_r);
        host.grids.data_z = UniformGridData::from_bounds(
            {inp.min_z, inp.max_z}, inp.num_grid_z);

        auto fieldmap = make_builder(&host.fieldmap);
        fieldmap.reserve(inp.field_z.size());
        for (auto i : range(inp.field_z.size()))
        {
            // Save field vector
            MapFieldElement el;
            el.value_z = inp.field_z[i];
            el.value_r = inp.field_r[i];
            fieldmap.push_back(el);
        }

        host.options = inp.driver_options;
        return host;
    }();

    // Move to mirrored data, copying to device
    mirror_ = CollectionMirror<RZMapFieldParamsData>{std::move(host_data)};
    CELER_ENSURE(this->mirror_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
