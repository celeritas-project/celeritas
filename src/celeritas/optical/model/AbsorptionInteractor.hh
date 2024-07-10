//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/AbsorptionInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/optical/Interaction.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Sample optical absorption interaction.
 *
 * Absorption rate is governed by sampling its mean free path in the action
 * loop. The interactor simply returns an interaction saying the optical
 * photon has been absorbed.
 */
class AbsorptionInteractor
{
  public:
    //! Sample an interaction (no RNG needed)
    inline CELER_FUNCTION Interaction operator()() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Sample an absorption interaction.
 */
CELER_FUNCTION Interaction AbsorptionInteractor::operator()() const
{
    return Interaction::from_absorption();
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
