//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/OpticalRayleighExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 */
struct OpticalRayleighExecutor
{
    inline CELER_FUNCTION OpticalInteraction operator()(OpticalTrackView const&)
    {
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
