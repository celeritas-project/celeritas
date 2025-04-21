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
class CartMapFieldParamsImpl;
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
    // TODO: move definition
    using real_type = float;
    using Input = CartMapFieldInput;
    //@}

  public:
    // Construct with a magnetic field map
    explicit CartMapFieldParams(Input const& inp);

    ~CartMapFieldParams();

    //! Access field map data on the host
    HostRef const& host_ref() const final;

    //! Access field map data on the device
    DeviceRef const& device_ref() const final;

  private:
    std::unique_ptr<CartMapFieldParamsImpl> impl_;
};

#if !CELERITAS_USE_COVFIE
inline CartMapFieldParams::CartMapFieldParams(Input const&)
{
    CELER_NOT_CONFIGURED("Covfie");
}

inline CartMapFieldParams::~CartMapFieldParams()
{
    CELER_NOT_CONFIGURED("Covfie");
}

//! Access field map data on the host
inline auto CartMapFieldParams::host_ref() const -> HostRef const&
{
    CELER_NOT_CONFIGURED("Covfie");
}

//! Access field map data on the device
inline auto CartMapFieldParams::device_ref() const -> DeviceRef const&
{
    CELER_NOT_CONFIGURED("Covfie");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
