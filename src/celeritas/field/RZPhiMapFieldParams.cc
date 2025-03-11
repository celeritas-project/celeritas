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
#include "corecel/Constants.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/UniformGridData.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/Units.hh"

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
    CELER_VALIDATE(inp.num_grid_z >= 2,
                   << "invalid field parameter (num_grid_z=" << inp.num_grid_z
                   << ")");
    CELER_VALIDATE(inp.num_grid_r >= 2,
                   << "invalid field parameter (num_grid_r=" << inp.num_grid_r
                   << ")");
    CELER_VALIDATE(inp.num_grid_phi >= 2,
                   << "invalid field parameter (num_grid_phi="
                   << inp.num_grid_phi << ")");
    CELER_VALIDATE(inp.min_r >= 0,
                   << "invalid field parameter (min_r=" << inp.min_r << ")");
    CELER_VALIDATE(inp.max_r > inp.min_r,
                   << "invalid field parameter (max_r=" << inp.max_r
                   << " <= min_r= " << inp.min_r << ")");
    CELER_VALIDATE(inp.max_z > inp.min_z,
                   << "invalid field parameter (max_z=" << inp.max_z
                   << " <= min_z= " << inp.min_z << ")");
    CELER_VALIDATE(inp.max_phi > inp.min_phi,
                   << "invalid field parameter (max_phi=" << inp.max_phi
                   << " <= min_phi= " << inp.min_phi << ")");

    CELER_VALIDATE(inp.field_z.size()
                       == inp.num_grid_z * inp.num_grid_r * inp.num_grid_phi,
                   << "invalid field length (field_z size="
                   << inp.field_z.size() << "): should be "
                   << inp.num_grid_z * inp.num_grid_r * inp.num_grid_phi);
    CELER_VALIDATE(
        inp.field_r.size() == inp.field_z.size(),
        << "invalid field length (field_r size=" << inp.field_r.size()
        << "): should be " << inp.field_z.size());
    CELER_VALIDATE(
        inp.field_phi.size() == inp.field_z.size(),
        << "invalid field length (field_phi size=" << inp.field_phi.size()
        << "): should be " << inp.field_z.size());

    // Throw a runtime error if any driver options are invalid
    validate_input(inp.driver_options);

    auto host_data = [&inp] {
        HostVal<RZPhiMapFieldParamsData> host;

        host.grids.data_r = UniformGridData::from_bounds(
            inp.min_r, inp.max_r, inp.num_grid_r);
        host.grids.data_z = UniformGridData::from_bounds(
            inp.min_z, inp.max_z, inp.num_grid_z);

        // For phi, ensure we're creating a periodic grid
        // If the input specifies a full circle, adjust the grid to handle
        // periodicity

        bool is_full_circle = soft_zero(
            std::fabs((inp.max_phi - inp.min_phi) - 2. * constants::pi));
        if (is_full_circle)
        {
            // For a full circle, we need one fewer phi point since phi=0 and
            // phi=2π are the same Create a grid from [min_phi, max_phi) where
            // the last point is just before max_phi
            if (inp.num_grid_phi > 2)
            {
                // Adjust delta to ensure we don't include the duplicate point
                // at max_phi
                real_type delta_phi = (inp.max_phi - inp.min_phi)
                                      / (inp.num_grid_phi - 1);
                real_type adjusted_max_phi
                    = inp.max_phi - delta_phi / 100;  // Slightly less than
                                                      // max_phi

                host.grids.data_phi = UniformGridData::from_bounds(
                    inp.min_phi, adjusted_max_phi, inp.num_grid_phi - 1);
            }
            else
            {
                // If we have too few points, just use the regular grid
                host.grids.data_phi = UniformGridData::from_bounds(
                    inp.min_phi, inp.max_phi, inp.num_grid_phi);
            }
        }
        else
        {
            // For partial circles, use the specified bounds as is
            host.grids.data_phi = UniformGridData::from_bounds(
                inp.min_phi, inp.max_phi, inp.num_grid_phi);
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