//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZPhiMapFieldParams.cc
//---------------------------------------------------------------------------//
#include "RZPhiMapFieldParams.hh"

#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/SoftEqual.hh"

#include "RZPhiMapField.hh"
#include "RZPhiMapFieldData.hh"
#include "RZPhiMapFieldInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a user-defined field map.
 */
RZPhiMapFieldParams::RZPhiMapFieldParams(RZPhiMapFieldInput const& inp)
{
    CELER_VALIDATE(
        inp.grid_z.size() >= 2,
        << "invalid field parameter (num_grid_z=" << inp.grid_z.size() << ")");
    CELER_VALIDATE(
        inp.grid_r.size() >= 2,
        << "invalid field parameter (num_grid_r=" << inp.grid_r.size() << ")");
    CELER_VALIDATE(inp.grid_phi.size() >= 2,
                   << "invalid field parameter (num_grid_phi="
                   << inp.grid_phi.size() << ")");
    CELER_VALIDATE(inp.field_r.front() >= 0,
                   << "invalid field parameter (min_r=" << inp.field_r.front()
                   << ")");
    CELER_VALIDATE(inp.field_r.back() > inp.field_r.front(),
                   << "invalid field parameter (max_r=" << inp.field_r.back()
                   << " <= min_r= " << inp.field_r.front() << ")");
    CELER_VALIDATE(inp.field_z.back() > inp.field_z.front(),
                   << "invalid field parameter (max_z=" << inp.field_z.back()
                   << " <= min_z= " << inp.field_z.front() << ")");
    CELER_VALIDATE(
        inp.field_phi.back() > inp.field_z.front(),
        << "invalid field parameter (max_phi=" << inp.field_phi.back()
        << " <= min_phi= " << inp.field_phi.front() << ")");

    CELER_VALIDATE(
        inp.field_z.size()
            == inp.grid_z.size() * inp.grid_r.size() * inp.grid_phi.size(),
        << "invalid field length (field_z size=" << inp.field_z.size()
        << "): should be "
        << inp.grid_z.size() * inp.grid_r.size() * inp.grid_phi.size());
    CELER_VALIDATE(
        inp.field_r.size() == inp.field_z.size(),
        << "invalid field length (field_r size=" << inp.field_r.size()
        << "): should be " << inp.field_z.size());
    CELER_VALIDATE(
        inp.field_phi.size() == inp.field_z.size(),
        << "invalid field length (field_phi size=" << inp.field_phi.size()
        << "): should be " << inp.field_z.size());
    CELER_VALIDATE(soft_zero(inp.grid_phi.front().value()),
                   << "Phi grid must be a complete circle (grid_phi min="
                   << inp.grid_phi.front().value() << "): should be 0");
    CELER_VALIDATE(soft_equal(1.0, inp.grid_phi.back().value()),
                   << "Phi grid must be a complete circle (grid_phi max="
                   << inp.grid_phi.back().value() << "): should be 1");

    // Throw a runtime error if any driver options are invalid
    validate_input(inp.driver_options);

    auto host_data = [&inp] {
        HostVal<RZPhiMapFieldParamsData> host;
        {
            auto builder = make_builder(&host.grids.z);
            builder.insert_back(inp.grid_z.begin(), inp.grid_z.end());
        }
        {
            auto builder = make_builder(&host.grids.r);
            builder.insert_back(inp.grid_r.begin(), inp.grid_r.end());
        }
        {
            auto builder = make_builder(&host.grids.phi);
            builder.insert_back(inp.grid_phi.begin(), inp.grid_phi.end());
        }

        auto fieldmap = make_builder(&host.fieldmap);
        fieldmap.reserve(inp.field_z.size());
        for (auto i : range(inp.field_z.size()))
        {
            // Save field vector
            RZPhiMapElement el;
            el.value_z = inp.field_z[i];
            el.value_r = inp.field_r[i];
            el.value_phi = inp.field_phi[i];
            fieldmap.push_back(el);
        }

        host.options = inp.driver_options;
        return host;
    }();

    // Move to mirrored data, copying to device
    mirror_ = CollectionMirror<RZPhiMapFieldParamsData>{std::move(host_data)};
    CELER_ENSURE(this->mirror_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas