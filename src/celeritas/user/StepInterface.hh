//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

#include "StepData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Callback class to gather and process data from many tracks at a single step.
 */
class StepInterface
{
  public:
    //@{
    //! \name Type aliases
    using StateHostRef   = HostRef<StepStateData>;
    using StateDeviceRef = DeviceRef<StepStateData>;
    //@}

  public:
    //! Process CPU-generated hit data
    virtual void operator()(StateHostRef const&) = 0;

    //! Process device-generated hit data
    virtual void operator()(StateDeviceRef const&) = 0;

  protected:
    // Protected destructor prevents deletion of pointer-to-interface
    ~StepInterface() = default;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
