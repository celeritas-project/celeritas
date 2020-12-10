//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Secondary.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"
#include "ParticleDef.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * New particle created via an Interaction.
 *
 * It will be converted into a "track initializer" using the parent track's
 * information.
 */
struct Secondary
{
    ParticleDefId    def_id;    //!< New particle type
    units::MevEnergy energy;    //!< New kinetic energy
    Real3            direction; //!< New direction

    // Default to invalid state
    CELER_FUNCTION
    Secondary() : def_id(ParticleDefId{}), energy(zero_quantity()) {}

    // Whether the secondary survived cutoffs
    explicit inline CELER_FUNCTION operator bool() const;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Whether the Secondary succeeded.
 */
CELER_FUNCTION Secondary::operator bool() const
{
    return static_cast<bool>(this->def_id);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
