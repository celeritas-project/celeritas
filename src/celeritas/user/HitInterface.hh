//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/HitInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

#include "HitData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Callback class to gather and process data from hits at a state point.
 */
class HitInterface
{
  public:
    //@{
    //! \name Type aliases
    using StateHostRef   = HostRef<HitStateData>;
    using StateDeviceRef = DeviceRef<HitStateData>;
    //@}

  public:
    //! Process CPU-generated hit data
    virtual void operator()(StateHostRef const&) = 0;

    //! Process device-generated hit data
    virtual void operator()(StateDeviceRef const&) = 0;

  protected:
    // Protected destructor prevents deletion of pointer-to-interface
    ~HitInterface() = default;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
