//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartMapFieldParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/data/ParamsDataInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct CartMapFieldInput;
template<Ownership W, MemSpace M>
struct CartMapFieldParamsData;

//---------------------------------------------------------------------------//
/*!
 * Set up a 3D CartMapFieldParams.
 *
 * The input values are in the native unit system.
 */
class CartMapFieldParams final
    : public ParamsDataInterface<CartMapFieldParamsData>
{
  public:
    //@{
    //! \name Type aliases
    using Input = CartMapFieldInput;
    //@}

  public:
    // Construct with a magnetic field map
    explicit CartMapFieldParams(Input const& inp);

    //! Access field map data on the host
    HostRef const& host_ref() const final;

    //! Access field map data on the device
    DeviceRef const& device_ref() const final;

  private:
    struct Impl;
    struct ImplDeleter
    {
        void operator()(Impl*) const noexcept;
    };
    std::unique_ptr<Impl, ImplDeleter> impl_;
};

#if !(CELERITAS_USE_COVFIE || __DOXYGEN__)

struct CartMapFieldParams::Impl
{
};

inline void CartMapFieldParams::ImplDeleter::operator()(Impl*) const noexcept
{
    CELER_UNREACHABLE;
}

inline CartMapFieldParams::CartMapFieldParams(Input const&)
{
    CELER_NOT_CONFIGURED("covfie");
}

//! Access field map data on the host
inline auto CartMapFieldParams::host_ref() const -> HostRef const&
{
    CELER_NOT_CONFIGURED("covfie");
}

//! Access field map data on the device
inline auto CartMapFieldParams::device_ref() const -> DeviceRef const&
{
    CELER_NOT_CONFIGURED("covfie");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
