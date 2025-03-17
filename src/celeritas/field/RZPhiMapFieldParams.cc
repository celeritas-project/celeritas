//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZPhiMapFieldParams.cc
//---------------------------------------------------------------------------//
#include "RZPhiMapFieldParams.hh"

#include <algorithm>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/Types.hh"
#include "celeritas/field/RZPhiMapField.hh"

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
    CELER_VALIDATE(inp.grid_r.front() >= 0,
                   << "invalid field parameter (min_r=" << inp.grid_r.front()
                   << ")");
    CELER_VALIDATE(inp.grid_r.back() > inp.grid_r.front(),
                   << "invalid field parameter (max_r=" << inp.grid_r.back()
                   << " <= min_r= " << inp.grid_r.front() << ")");
    CELER_VALIDATE(inp.grid_z.back() > inp.grid_z.front(),
                   << "invalid field parameter (max_z=" << inp.grid_z.back()
                   << " <= min_z= " << inp.grid_z.front() << ")");
    CELER_VALIDATE(
        inp.grid_phi.back() > inp.grid_phi.front(),
        << "invalid field parameter (max_phi=" << inp.grid_phi.back().value()
        << " <= min_phi= " << inp.grid_phi.front().value() << ")");

    CELER_VALIDATE(
        inp.field.size()
            == static_cast<size_type>(CylAxis::size_) * inp.grid_z.size()
                   * inp.grid_r.size() * inp.grid_phi.size(),
        << "invalid field length (field size=" << inp.field.size()
        << "): should be "
        << 3 * inp.grid_z.size() * inp.grid_r.size() * inp.grid_phi.size());
    CELER_VALIDATE(soft_zero(inp.grid_phi.front().value()),
                   << "Phi grid must be a complete circle (grid_phi min="
                   << inp.grid_phi.front().value() << "): should be 0");
    CELER_VALIDATE(soft_equal(real_type{1}, inp.grid_phi.back().value()),
                   << "Phi grid must be a complete circle (grid_phi max="
                   << inp.grid_phi.back().value() << "): should be 1");

    // Throw a runtime error if any driver options are invalid
    validate_input(inp.driver_options);

    auto host_data = [&inp] {
        HostVal<RZPhiMapFieldParamsData> host;

        host.grids.grid_size[CylAxis::Phi] = inp.grid_phi.size();
        host.grids.grid_size[CylAxis::R] = inp.grid_r.size();
        host.grids.grid_size[CylAxis::Z] = inp.grid_z.size();

        auto grid = make_builder(&host.grids.storage);
        grid.reserve(inp.grid_phi.size() + inp.grid_r.size()
                     + inp.grid_z.size());
        std::transform(inp.grid_phi.cbegin(),
                       inp.grid_phi.cend(),
                       std::back_inserter(grid),
                       [](auto const& val) { return val.value(); });
        host.grids.axes[CylAxis::Phi] = ItemRange<real_type>{grid.size_id()};
        host.grids.axes[CylAxis::R]
            = grid.insert_back(inp.grid_r.begin(), inp.grid_r.end());
        host.grids.axes[CylAxis::Z]
            = grid.insert_back(inp.grid_z.begin(), inp.grid_z.end());

        auto fieldmap = make_builder(&host.fieldmap);
        fieldmap.reserve(inp.field.size());
        for (auto i :
             range(inp.grid_z.size() * inp.grid_r.size() * inp.grid_phi.size()))
        {
            // Save field vector
            fieldmap.push_back(
                [&, idx = i * static_cast<size_type>(CylAxis::size_)] {
                    EnumArray<CylAxis, real_type> el;
                    for (auto axis : range(CylAxis::size_))
                    {
                        el[axis]
                            = inp.field[idx + static_cast<size_type>(axis)];
                    }
                    return el;
                }());
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