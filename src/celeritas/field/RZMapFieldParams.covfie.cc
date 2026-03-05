//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldParams.covfie.cc
//---------------------------------------------------------------------------//
#include "RZMapFieldParams.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "geocel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/field/RZMapFieldData.hh"
#include "celeritas/field/RZMapFieldInput.hh"

#include "RZMapField.covfie.hh"

#include "detail/CovfieRZFieldTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct RZMapFieldParams::Impl
{
    explicit Impl(Input const& inp)
        : host_{[&inp] {
            CELER_VALIDATE(inp.num_grid_z >= 2,
                           << "invalid field parameter (num_grid_z="
                           << inp.num_grid_z << ")");
            CELER_VALIDATE(inp.num_grid_r >= 2,
                           << "invalid field parameter (num_grid_r="
                           << inp.num_grid_r << ")");
            CELER_VALIDATE(inp.min_r >= 0,
                           << "invalid field parameter (min_r=" << inp.min_r
                           << ")");
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
            validate_input(inp.driver_options);

            HostVal<RZMapFieldParamsData> host;

            using builder_t
                = detail::CovfieRZFieldTraits<MemSpace::host>::builder_t;

            builder_t builder{covfie::make_parameter_pack(
                builder_t::backend_t::configuration_t{inp.num_grid_r,
                                                      inp.num_grid_z})};
            builder_t::view_t builder_view{builder};

            // Fill the covfie field data
            // Input layout is [Z][R]: index = iz * num_grid_r + ir
            for (auto iz : range(inp.num_grid_z))
            {
                for (auto ir : range(inp.num_grid_r))
                {
                    auto idx = iz * inp.num_grid_r + ir;
                    builder_view.at(ir, iz)
                        = {static_cast<float>(inp.field_r[idx]),
                           static_cast<float>(inp.field_z[idx])};
                }
            }

            using field_real_type = RZMapField::real_type;
            // Shift world coordinates so the grid minimum maps to zero.
            auto affine_translate = covfie::algebra::affine<2>::translation(
                static_cast<field_real_type>(-inp.min_r),
                static_cast<field_real_type>(-inp.min_z));

            // Scale world units to index units so max maps to (num - 1).
            auto affine_scale = covfie::algebra::affine<2>::scaling(
                static_cast<field_real_type>((inp.num_grid_r - 1)
                                             / (inp.max_r - inp.min_r)),
                static_cast<field_real_type>((inp.num_grid_z - 1)
                                             / (inp.max_z - inp.min_z)));

            using traits_t = detail::CovfieRZFieldTraits<MemSpace::host>;
            using field_t = typename traits_t::field_t;
            using clamp_config_t =
                typename traits_t::clamped_t::configuration_t;
            using clamp_vec_t = decltype(clamp_config_t{}.min);
            using clamp_scalar_t = typename clamp_vec_t::value_type;

            auto clamp_max = [](size_type n) {
                auto const max = static_cast<clamp_scalar_t>(n - 1);
                return std::nextafter(
                    max, -std::numeric_limits<clamp_scalar_t>::infinity());
            };

            // Clamp in index space before interpolation: keep coords in-bounds
            // and avoid the (n-1) corner that would request n in linear.
            clamp_config_t clamp_config{
                clamp_vec_t{static_cast<clamp_scalar_t>(0),
                            static_cast<clamp_scalar_t>(0)},
                clamp_vec_t{clamp_max(inp.num_grid_r),
                            clamp_max(inp.num_grid_z)}};

            host.field = std::make_unique<field_t>(covfie::make_parameter_pack(
                field_t::backend_t::configuration_t(affine_scale
                                                    * affine_translate),
                std::move(clamp_config),
                typename traits_t::interp_t::configuration_t{},
                builder.backend()));
            host.options = inp.driver_options;
            host.min_r = inp.min_r;
            host.max_r = inp.max_r;
            host.min_z = inp.min_z;
            host.max_z = inp.max_z;
            return host;
        }()}
        , host_ref_{
              {host_.options, host_.min_r, host_.max_r, host_.min_z, host_.max_z},
              *host_.field}
    {
        if (celeritas::device())
        {
            device_ = host_;
            device_ref_ = device_;
            CELER_ENSURE(static_cast<bool>(device_)
                         && static_cast<bool>(device_ref_));
        }
        CELER_LOG(debug) << "Constructed RZMapField with covfie 2D "
                            "interpolation ("
                         << inp.num_grid_r << "x" << inp.num_grid_z
                         << " grid)";
        if (celeritas::device())
        {
            CELER_LOG(debug) << "Copied RZMapField covfie data to device";
        }
        CELER_ENSURE(static_cast<bool>(host_) && static_cast<bool>(host_ref_));
    }

    HostVal<RZMapFieldParamsData> host_;
    HostCRef<RZMapFieldParamsData> host_ref_;
    RZMapFieldParamsData<Ownership::value, MemSpace::device> device_;
    DeviceRef device_ref_;
};

//---------------------------------------------------------------------------//
/*!
 * Custom deleter for the implementation.
 */
void RZMapFieldParams::ImplDeleter::operator()(Impl* impl) const noexcept
{
    delete impl;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a user-defined field map.
 */
RZMapFieldParams::RZMapFieldParams(Input const& inp)
    : impl_{std::unique_ptr<Impl, ImplDeleter>(new Impl{inp})}
{
}

//---------------------------------------------------------------------------//
//! Access field map data on the host
auto RZMapFieldParams::host_ref() const -> HostRef const&
{
    return impl_->host_ref_;
}

//---------------------------------------------------------------------------//
//! Access field map data on the device
auto RZMapFieldParams::device_ref() const -> DeviceRef const&
{
    return impl_->device_ref_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
