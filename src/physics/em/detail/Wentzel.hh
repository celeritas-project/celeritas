//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file WentzelInteractorPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
struct WentzelInteractorPointers
{
    //! ID of an electron
    ParticleId electron_id;
    //! ID of a gamma
    ParticleId gamma_id;
    // XXX additional data

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return electron_id && gamma_id; // XXX
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
