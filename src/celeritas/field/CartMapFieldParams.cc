//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartMapFieldParams.cc
//---------------------------------------------------------------------------//
#include "CartMapFieldParams.hh"

#include <algorithm>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/HyperslabIndexer.hh"
#include "celeritas/Types.hh"
#include "celeritas/field/CartMapFieldData.hh"
#include "celeritas/field/detail/CovfieFieldType.hh"

#include "CartMapFieldData.hh"
#include "CartMapFieldInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a user-defined field map.
 */
CartMapFieldParams::CartMapFieldParams(Input const& inp)
{
    auto host_data = [&inp] {
        HostVal<CartMapFieldParamsData> host;

        Array<size_type, 4> const dims{inp.num_x,
                                       inp.num_y,
                                       inp.num_z,
                                       static_cast<size_type>(CartAxis::size_)};
        HyperslabIndexer const flat_index{dims};

        covfie::algebra::affine<3> affine_translate
            = covfie::algebra::affine<3>::translation(
                -inp.min_x, -inp.min_y, -inp.min_z);

        covfie::algebra::affine<3> affine_scale
            = covfie::algebra::affine<3>::scaling(
                (inp.num_x - 1) / (inp.max_x - inp.min_x),
                (inp.num_y - 1) / (inp.max_y - inp.min_y),
                (inp.num_z - 1) / (inp.max_z - inp.min_z));

        using field_t = CovfieFieldTrait<MemSpace::host>::field_t;

        field_t field{covfie::make_parameter_pack(
            field_t::backend_t::configuration_t(affine_scale * affine_translate),
            field_t::backend_t::backend_t::configuration_t{},
            field_t::backend_t::backend_t::backend_t::configuration_t{
                inp.num_x, inp.num_y, inp.num_z})};

        field_t::view_t field_view{field};
        // fill the covfie field data
        for (auto ix : range(inp.num_x))
        {
            for (auto iy : range(inp.num_y))
            {
                for (auto iz : range(inp.num_z))
                {
                    auto* fv = field_view.at(ix, iy, iz).begin();
                    auto* finp = inp.field.data() + flat_index(ix, iy, iz, 0);
                    std::copy(finp,
                              finp + static_cast<size_type>(CartAxis::size_),
                              fv);
                }
            }
        }

        host.field = std::move(field);
        host.options = inp.driver_options;
        return host;
    }();

    // Move to mirrored data, copying to device
    mirror_ = CollectionMirror<CartMapFieldParamsData>{std::move(host_data)};
    CELER_ENSURE(this->mirror_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
