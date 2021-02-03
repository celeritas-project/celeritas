//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BhabhaEnergyDistribution.hh
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
class BhabhaEnergyDistribution
{
  public:
    // Construct with data from MollerBhabhaInteractor
    inline CELER_FUNCTION
    BhabhaEnergyDistribution(const MollerBhabhaPointers& shared,
                             const real_type             inc_energy);

    // Sample the exiting energy
    template<class Engine>
    inline CELER_FUNCTION real_type operator()(Engine& rng);

  private:
    // Electron mass * c^2 [MeV]
    real_type electron_mass_c_sq_;
    // Electron incident energy [MeV]
    real_type inc_energy_;
    // Total energy of the incident particle
    real_type total_energy_;
    // Maximum energy fraction transferred to free electron
    real_type max_energy_fraction_;
    // Maximum energy fraction transferred to free electron
    real_type min_energy_fraction_;
    // Sampling parameter
    real_type gamma_;

  private:
    // Calculate value for probability or rejection functions f and g
    inline CELER_FUNCTION real_type calc_f_g(real_type epsilon_min,
                                             real_type epsilon_max);
}; // namespace BhabhaEnergyDistribution

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "BhabhaEnergyDistribution.i.hh"
