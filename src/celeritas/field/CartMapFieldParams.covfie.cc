//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartMapFieldParams.covfie.cc
//---------------------------------------------------------------------------//
#include "CartMapFieldParams.hh"

#include <algorithm>
#include <memory>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/HyperslabIndexer.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/sys/Device.hh"
#include "geocel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/field/CartMapFieldData.hh"
#include "celeritas/field/CartMapFieldInput.hh"

#include "CartMapField.covfie.hh"

#include "detail/CovfieFieldTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct CartMapFieldParams::Impl
{
  public:
    using HostRef = ParamsDataInterface<CartMapFieldParamsData>::HostRef;
    using DeviceRef = ParamsDataInterface<CartMapFieldParamsData>::DeviceRef;
    using Input = CartMapFieldParams::Input;

    Impl(Input const& inp)
        : host_{[&inp] {
            HostVal<CartMapFieldParamsData> host;

            Array<size_type, 4> const dims{inp.num_x,
                                           inp.num_y,
                                           inp.num_z,
                                           static_cast<size_type>(Axis::size_)};
            HyperslabIndexer const flat_index{dims};

            using builder_t
                = detail::CovfieFieldTraits<MemSpace::host>::builder_t;

            builder_t builder{covfie::make_parameter_pack(
                builder_t::backend_t::configuration_t{
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
                        auto* finp = inp.field.data()
                                     + flat_index(ix, iy, iz, 0);
                        std::copy(finp,
                                  finp + static_cast<size_type>(Axis::size_),
                                  fv);
                    }
                }
            }

            using field_real_type = CartMapField::real_type;
            auto affine_translate = covfie::algebra::affine<3>::translation(
                static_cast<field_real_type>(-inp.min_x),
                static_cast<field_real_type>(-inp.min_y),
                static_cast<field_real_type>(-inp.min_z));

            auto affine_scale = covfie::algebra::affine<3>::scaling(
                static_cast<field_real_type>((inp.num_x - 1)
                                             / (inp.max_x - inp.min_x)),
                static_cast<field_real_type>((inp.num_y - 1)
                                             / (inp.max_y - inp.min_y)),
                static_cast<field_real_type>((inp.num_z - 1)
                                             / (inp.max_z - inp.min_z)));

            using field_t = detail::CovfieFieldTraits<MemSpace::host>::field_t;
            host.field = std::make_unique<field_t>(covfie::make_parameter_pack(
                field_t::backend_t::configuration_t(affine_scale
                                                    * affine_translate),
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

    HostVal<CartMapFieldParamsData> host_;
    HostCRef<CartMapFieldParamsData> host_ref_;
    CartMapFieldParamsData<Ownership::value, MemSpace::device> device_;
    DeviceRef device_ref_;
};

//---------------------------------------------------------------------------//
/*!
 * Custom deleter for the implementation.
 */
void CartMapFieldParams::ImplDeleter::operator()(Impl* impl) noexcept
{
    delete impl;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a user-defined field map.
 */
CartMapFieldParams::CartMapFieldParams(Input const& inp)
    : impl_{std::unique_ptr<Impl, ImplDeleter>(new Impl{inp})}

{
}

//---------------------------------------------------------------------------//
//! Access field map data on the host
auto CartMapFieldParams::host_ref() const -> HostRef const&
{
    return impl_->host_ref_;
}

//---------------------------------------------------------------------------//
//! Access field map data on the device
auto CartMapFieldParams::device_ref() const -> DeviceRef const&
{
    return impl_->device_ref_;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
