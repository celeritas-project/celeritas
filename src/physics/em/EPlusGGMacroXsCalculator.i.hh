//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGGMacroXsCalculator.i.hh
//---------------------------------------------------------------------------//

#include "base/Algorithms.hh"
#include "base/Assert.hh"
#include "base/Constants.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with material.
 */
CELER_FUNCTION EPlusGGMacroXsCalculator::EPlusGGMacroXsCalculator(
    const EPlusGGPointers& shared, const MaterialView& material)
    : electron_mass_(shared.electron_mass)
    , electron_density_(material.electron_density())
{
}

//---------------------------------------------------------------------------//
/*!
 * The Heitler formula (section 10.3.2 of the Geant4 Physics Reference Manual,
 * Release 10.6) is used to compute the macroscopic cross section for positron
 * annihilation on the fly at the given energy.
 */
CELER_FUNCTION real_type
EPlusGGMacroXsCalculator::operator()(MevEnergy energy) const
{
    using constants::pi;
    using constants::re_electron;

    energy                 = max(MevEnergy{1.e-6}, energy);
    const real_type gamma  = energy.value() / electron_mass_;
    const real_type g1     = gamma + 1.;
    const real_type g2     = gamma * (gamma + 2.);
    real_type       result = electron_density_ * pi * re_electron * re_electron
                       * ((g1 * (g1 + 4) + 1.) * std::log(g1 + std::sqrt(g2))
                          - (g1 + 3.) * std::sqrt(g2))
                       / (g2 * (g1 + 1.));

    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
