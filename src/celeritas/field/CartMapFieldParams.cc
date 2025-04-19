//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartMapFieldParams.cc
//---------------------------------------------------------------------------//
#include "CartMapFieldParams.hh"

#include <algorithm>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/HyperslabIndexer.hh"
#include "corecel/sys/Device.hh"
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
    : host_{[&inp] {
        HostVal<CartMapFieldParamsData> host;

        Array<size_type, 4> const dims{inp.num_x,
                                       inp.num_y,
                                       inp.num_z,
                                       static_cast<size_type>(CartAxis::size_)};
        HyperslabIndexer const flat_index{dims};

        using builder_t = CovfieFieldTrait<MemSpace::host>::builder_t;

        builder_t builder{
            covfie::make_parameter_pack(builder_t::backend_t::configuration_t{
                inp.num_x, inp.num_y, inp.num_z})};
        builder_t::view_t builder_view{builder};
        // fill the covfie field data
        for (auto ix : range(inp.num_x))
        {
            for (auto iy : range(inp.num_y))
            {
                for (auto iz : range(inp.num_z))
                {
                    auto* fv = builder_view.at(ix, iy, iz).begin();
                    auto* finp = inp.field.data() + flat_index(ix, iy, iz, 0);
                    std::copy(finp,
                              finp + static_cast<size_type>(CartAxis::size_),
                              fv);
                }
            }
        }

        auto affine_translate = covfie::algebra::affine<3>::translation(
            -inp.min_x, -inp.min_y, -inp.min_z);

        auto affine_scale = covfie::algebra::affine<3>::scaling(
            (inp.num_x - 1) / (inp.max_x - inp.min_x),
            (inp.num_y - 1) / (inp.max_y - inp.min_y),
            (inp.num_z - 1) / (inp.max_z - inp.min_z));

        using field_t = CovfieFieldTrait<MemSpace::host>::field_t;
        host.field = std::make_unique<field_t>(covfie::make_parameter_pack(
            field_t::backend_t::configuration_t(affine_scale * affine_translate),
            field_t::backend_t::backend_t::configuration_t{},
            builder.backend()));
        host.options = inp.driver_options;
        return host;
    }()}
    , host_ref_{{host_.options}, *host_.field}
{
    if (celeritas::device())
    {
        device_ = host_;
        device_ref_ = device_;
        CELER_ENSURE(static_cast<bool>(device_)
                     && static_cast<bool>(device_ref_));
    }
    CELER_ENSURE(static_cast<bool>(host_) && static_cast<bool>(host_ref_));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
