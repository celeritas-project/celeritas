//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerEnergyDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "MollerBhabha.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for \c MollerBhabhaInteractor .
 *
 * Sample the exiting energy for Moller scattering.
 */
class MollerEnergyDistribution
{
  public:
    // Construct with data from MollerBhabhaInteractor
    inline CELER_FUNCTION MollerEnergyDistribution(real_type electron_mass_c_sq,
                                                   real_type min_valid_energy,
                                                   real_type inc_energy);

    // Sample the exiting energy
    template<class Engine>
    inline CELER_FUNCTION real_type operator()(Engine& rng);

  private:
    // Electron incident energy [MeV]
    real_type inc_energy_;
    // Total energy of the incident particle [MeV]
    real_type total_energy_;
    // Minimum energy fraction transferred to free electron [MeV]
    real_type min_energy_fraction_;
    // Sampling parameter
    real_type gamma_;

  private:
    // Helper function for calculating rejection function g
    inline CELER_FUNCTION real_type calc_g_fraction(real_type epsilon);
    // Maximum energy fraction transferred to free electron [MeV]
    static CELER_CONSTEXPR_FUNCTION real_type max_energy_fraction()
    {
        return 0.5;
    }
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "MollerEnergyDistribution.i.hh"
