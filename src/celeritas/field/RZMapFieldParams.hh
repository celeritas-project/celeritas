//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Config.hh"

#include "corecel/data/ParamsDataInterface.hh"

#if !CELERITAS_USE_COVFIE
#    include "corecel/data/ParamsDataStore.hh"

#    include "RZMapFieldData.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
struct RZMapFieldInput;
template<Ownership W, MemSpace M>
struct RZMapFieldParamsData;

//---------------------------------------------------------------------------//
/*!
 * Set up a 2D RZMapFieldParams.
 *
 * The input values should be converted to the native unit system.
 */
class RZMapFieldParams final : public ParamsDataInterface<RZMapFieldParamsData>
{
  public:
    //@{
    //! \name Type aliases
    using Input = RZMapFieldInput;
    //@}

  public:
    // Construct with a magnetic field map
    explicit RZMapFieldParams(Input const& inp);

    //! Access field map data on the host
    HostRef const& host_ref() const final;

    //! Access field map data on the device
    DeviceRef const& device_ref() const final;

  private:
#if CELERITAS_USE_COVFIE
    struct Impl;
    struct ImplDeleter
    {
        void operator()(Impl*) const noexcept;
    };
    std::unique_ptr<Impl, ImplDeleter> impl_;
#else
    // Host/device storage and reference
    ParamsDataStore<RZMapFieldParamsData> mirror_;
#endif
};

#if !CELERITAS_USE_COVFIE
//! Access field map data on the host
inline auto RZMapFieldParams::host_ref() const -> HostRef const&
{
    return mirror_.host_ref();
}

//! Access field map data on the device
inline auto RZMapFieldParams::device_ref() const -> DeviceRef const&
{
    return mirror_.device_ref();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
