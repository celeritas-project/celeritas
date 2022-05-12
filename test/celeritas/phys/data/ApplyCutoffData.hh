//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ApplyCutoffData.hh
//---------------------------------------------------------------------------//
#pragma once
#include "celeritas/Types.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Model and particles IDs.
 */
struct ApplyCutoffIds
{
    celeritas::ActionId action;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(action);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data for the killer kernel.
 */
struct ApplyCutoffData
{
    //! Action ID used by interactor launch adapter
    ApplyCutoffIds ids;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(ids);
    }
};

using ApplyCutoffDeviceRef = ApplyCutoffData;
using ApplyCutoffHostRef   = ApplyCutoffData;

//---------------------------------------------------------------------------//
} // namespace celeritas_test
