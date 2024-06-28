//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/AbsorptionExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 */
struct AbsorptionExecutor
{
    inline CELER_FUNCTION Interaction
    operator()(OpticalTrackView const&)
    {
        return Interaction::from_absorption();
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
