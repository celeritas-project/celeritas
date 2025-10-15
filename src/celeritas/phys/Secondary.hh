//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/Secondary.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

#include "ParticleData.hh"

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
    ParticleId particle_id;  //!< New particle type
    units::MevEnergy energy;  //!< New kinetic energy
    Real3 direction;  //!< New direction

    //! Whether the secondary survived cutoffs
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(this->particle_id);
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
