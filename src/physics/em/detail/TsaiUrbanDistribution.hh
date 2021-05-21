//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TsaiUrbanDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Angular distribution for pair-production and bremsstrahlung processes.
 *
 * For pair-production, the polar angle of the electron (or positron) is
 * defined with respect to the direction of the parent photon. The energy-
 * angle distribution given by Tsai is quite complicated to
 * sample and can be approximated by a density function suggested by Urban.
 *
 * The angular distribution of the emitted photons is obtained from a
 * simplified formula based on the Tsai cross-section,
 * which is expected to become isotropic in the low energy limit.
 *
 * \note This performs the same sampling routine as in Geant4's
 * ModifiedTsai class, based on derivation from Tsai (Rev Mod Phys 49,421(1977)
 * and documented in section 6.5.2 (pair-production), and 10.2.1 and 10.2.4
 * (bremsstrahlung) of the Geant4 Physics Reference (release 10.6).
 */
class TsaiUrbanDistribution
{
    using Energy      = units::MevEnergy;
    using Mass        = units::MevMass;
    using result_type = real_type;

  public:
    // Construct with defaults
    inline CELER_FUNCTION
    TsaiUrbanDistribution(Energy inc_energy, Mass inc_mass);

    // Sample using the given random number generator
    template<class Engine>
    inline CELER_FUNCTION result_type operator()(Engine& rng);

  private:
    const Energy energy_;
    const Mass   mass_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "TsaiUrbanDistribution.i.hh"
