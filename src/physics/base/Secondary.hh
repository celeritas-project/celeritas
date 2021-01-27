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
    ParticleId       def_id{};                //!< New particle type
    units::MevEnergy energy{zero_quantity()}; //!< New kinetic energy
    Real3            direction;               //!< New direction

    //! Whether the secondary survived cutoffs
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(this->def_id);
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
