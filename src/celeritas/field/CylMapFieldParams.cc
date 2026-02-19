//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CylMapFieldParams.cc
//---------------------------------------------------------------------------//
#include "CylMapFieldParams.hh"

#include <algorithm>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/Types.hh"

#include "CylMapFieldData.hh"
#include "CylMapFieldInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a user-defined field map.
 */
CylMapFieldParams::CylMapFieldParams(CylMapFieldInput const& inp)
{
    CELER_VALIDATE(
        inp.grid_r.size() >= 2,
        << "invalid field parameter (num_grid_r=" << inp.grid_r.size() << ")");
    CELER_VALIDATE(inp.grid_phi.size() >= 2,
                   << "invalid field parameter (num_grid_phi="
                   << inp.grid_phi.size() << ")");
    CELER_VALIDATE(
        inp.grid_z.size() >= 2,
        << "invalid field parameter (num_grid_z=" << inp.grid_z.size() << ")");

    CELER_VALIDATE(inp.grid_r.back() > inp.grid_r.front(),
                   << "invalid field parameter (max_r=" << inp.grid_r.back()
                   << " <= min_r= " << inp.grid_r.front() << ")");
    CELER_VALIDATE(
        inp.grid_phi.back() > inp.grid_phi.front(),
        << "invalid field parameter (max_phi=" << inp.grid_phi.back().value()
        << " <= min_phi= " << inp.grid_phi.front().value() << ")");
    CELER_VALIDATE(inp.grid_z.back() > inp.grid_z.front(),
                   << "invalid field parameter (max_z=" << inp.grid_z.back()
                   << " <= min_z= " << inp.grid_z.front() << ")");

    CELER_VALIDATE(inp.grid_r.front() >= 0,
                   << "invalid field parameter (min_r=" << inp.grid_r.front()
                   << ")");
    CELER_VALIDATE(soft_zero(inp.grid_phi.front().value()),
                   << "Phi grid must be a complete circle (grid_phi min="
                   << inp.grid_phi.front().value() << "): should be 0");
    CELER_VALIDATE(
        soft_equal(celeritas::real_type{1}, inp.grid_phi.back().value()),
        << "Phi grid must be a complete circle (grid_phi max="
        << inp.grid_phi.back().value() << "): should be 1");

    CELER_VALIDATE(
        inp.field.size()
            == static_cast<size_type>(CylAxis::size_) * inp.grid_r.size()
                   * inp.grid_phi.size() * inp.grid_z.size(),
        << "invalid field length (field size=" << inp.field.size()
        << "): should be "
        << static_cast<size_type>(CylAxis::size_) * inp.grid_r.size()
                   * inp.grid_phi.size() * inp.grid_z.size());

    // Throw a runtime error if any driver options are invalid
    validate_input(inp.driver_options);

    auto host_data = [&inp] {
        HostVal<CylMapFieldParamsData> host;

        auto grid = make_builder(&host.grids.storage);
        grid.reserve(inp.grid_phi.size() + inp.grid_r.size()
                     + inp.grid_z.size());

        auto r_start = grid.size_id();
        std::transform(
            inp.grid_r.cbegin(),
            inp.grid_r.cend(),
            std::back_inserter(grid),
            [](auto const& val) { return static_cast<real_type>(val); });
        host.grids.axes[CylAxis::r]
            = ItemRange<real_type>{r_start, grid.size_id()};

        // Replace first and last phi grid values with exact zero and unity
        auto phi_start = grid.size_id();
        grid.push_back(0);
        std::transform(inp.grid_phi.cbegin() + 1,
                       inp.grid_phi.cend() - 1,
                       std::back_inserter(grid),
                       [](auto const& val) {
                           return static_cast<real_type>(val.value());
                       });
        grid.push_back(1);
        host.grids.axes[CylAxis::phi]
            = ItemRange<real_type>{phi_start, grid.size_id()};

        auto z_start = grid.size_id();
        std::transform(
            inp.grid_z.cbegin(),
            inp.grid_z.cend(),
            std::back_inserter(grid),
            [](auto const& val) { return static_cast<real_type>(val); });
        host.grids.axes[CylAxis::z]
            = ItemRange<real_type>{z_start, grid.size_id()};

        auto fieldmap = make_builder(&host.fieldmap);
        fieldmap.reserve(inp.field.size());
        for (auto i :
             range(inp.grid_r.size() * inp.grid_phi.size() * inp.grid_z.size()))
        {
            // Save field vector
            fieldmap.push_back(
                [idx = inp.field.cbegin()
                       + i * static_cast<size_type>(CylAxis::size_)] {
                    EnumArray<CylAxis, real_type> el;
                    std::transform(
                        idx,
                        idx + static_cast<size_type>(CylAxis::size_),
                        el.begin(),
                        [](auto val) { return static_cast<real_type>(val); });
                    return el;
                }());
        }

        host.options = inp.driver_options;
        return host;
    }();

    // Move to mirrored data, copying to device
    mirror_ = ParamsDataStore<CylMapFieldParamsData>{std::move(host_data)};
    CELER_ENSURE(this->mirror_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
