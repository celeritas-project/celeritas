//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/MuBremsDiffXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/mat/ElementView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate differential cross sections for muon bremsstrahlung.
 */
class MuBremsDiffXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using Mass = units::MevMass;
    //!@}

  public:
    // Construct with incident particle data and current element
    inline CELER_FUNCTION MuBremsDiffXsCalculator(ElementView const& element,
                                                  Energy inc_energy,
                                                  Mass inc_mass,
                                                  Mass electron_mass);

    // Compute cross section of exiting gamma energy
    inline CELER_FUNCTION real_type operator()(Energy energy);

  private:
    //// DATA ////

    // Shared problem data for the current element
    ElementView const& element_;
    // Energy of the incident particle
    real_type inc_energy_;
    // Mass of the incident particle
    real_type inc_mass_;
    // Total energy of the incident particle
    real_type total_energy_;
    // Mass of an electron
    real_type electron_mass_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with incident particle data and current element.
 */
CELER_FUNCTION
MuBremsDiffXsCalculator::MuBremsDiffXsCalculator(ElementView const& element,
                                                 Energy inc_energy,
                                                 Mass inc_mass,
                                                 Mass electron_mass)
    : element_(element)
    , inc_energy_(value_as<Energy>(inc_energy))
    , inc_mass_(value_as<Mass>(inc_mass))
    , total_energy_(inc_energy_ + inc_mass_)
    , electron_mass_(value_as<Mass>(electron_mass))
{
    CELER_EXPECT(inc_energy_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Compute the differential cross section per atom at the given photon energy.
 */
CELER_FUNCTION
real_type MuBremsDiffXsCalculator::operator()(Energy energy)
{
    CELER_EXPECT(energy > zero_quantity());

    if (value_as<Energy>(energy) >= inc_energy_)
    {
        return 0;
    }

    int const atomic_number = element_.atomic_number().unchecked_get();
    real_type const atomic_mass
        = value_as<units::AmuMass>(element_.atomic_mass());
    real_type const sqrt_e = std::sqrt(constants::euler);
    real_type const rel_energy_transfer = value_as<Energy>(energy)
                                          / total_energy_;
    real_type const inc_mass_sq = ipow<2>(inc_mass_);
    real_type const delta = real_type(0.5) * inc_mass_sq * rel_energy_transfer
                            / (total_energy_ - value_as<Energy>(energy));

    // TODO: precalculate these data per element
    real_type d_n_prime, b, b1;
    real_type const d_n = real_type(1.54)
                          * std::pow(atomic_mass, real_type(0.27));

    if (atomic_number == 1)
    {
        d_n_prime = d_n;
        b = real_type(202.4);
        b1 = 446;
    }
    else
    {
        d_n_prime = std::pow(d_n, 1 - real_type(1) / atomic_number);
        b = 183;
        b1 = 1429;
    }

    real_type const inv_cbrt_z = 1 / element_.cbrt_z();

    real_type const phi_n = clamp_to_nonneg(std::log(
        b * inv_cbrt_z * (inc_mass_ + delta * (d_n_prime * sqrt_e - 2))
        / (d_n_prime * (electron_mass_ + delta * sqrt_e * b * inv_cbrt_z))));

    real_type phi_e = 0;
    real_type const epsilon_max_prime
        = total_energy_
          / (1
             + real_type(0.5) * inc_mass_sq / (electron_mass_ * total_energy_));

    if (value_as<Energy>(energy) < epsilon_max_prime)
    {
        real_type const inv_cbrt_z_sq = ipow<2>(inv_cbrt_z);
        phi_e = clamp_to_nonneg(std::log(
            b1 * inv_cbrt_z_sq * inc_mass_
            / ((1 + delta * inc_mass_ / (ipow<2>(electron_mass_) * sqrt_e))
               * (electron_mass_ + delta * sqrt_e * b1 * inv_cbrt_z_sq))));
    }

    return 16 * constants::alpha_fine_structure * constants::na_avogadro
           * ipow<2>(electron_mass_ * constants::r_electron) * atomic_number
           * (atomic_number * phi_n + phi_e)
           * (1
              - rel_energy_transfer
                    * (1 - real_type(0.75) * rel_energy_transfer))
           / (3 * inc_mass_sq * value_as<Energy>(energy) * atomic_mass);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
