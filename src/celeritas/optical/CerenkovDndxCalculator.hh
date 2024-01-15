//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CerenkovDndxCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/grid/GenericCalculator.hh"
#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/grid/VectorUtils.hh"

#include "CerenkovData.hh"
#include "OpticalPropertyData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the mean number of Cerenkov photons produced per unit length.
 *
 * The average number of photons produced is given by
 * \f[
   dN = \frac{\alpha z^2}{\hbar c}\sin^2\theta d\epsilon dx = \frac{\alpha
 z^2}{\hbar c}\left(1 - \frac{1}{n^2\beta^2}\right) d\epsilon dx,
 * \f]
 * where \f$ n \f$ is the refractive index of the material, \f$ \epsilon \f$
 * is the photon energy, and \f$ \theta \f$ is the angle of the emitted photons
 * with respect to the incident particle direction, given by \f$ \cos\theta = 1
 * / (\beta n) \f$. Note that in a dispersive medium, the index of refraction
 * is an inreasing function of photon energy. The mean number of photons per
 * unit length is given by
 * \f[
   dN/dx = \frac{\alpha z^2}{\hbar
 c}\int_{\epsilon_\text{min}}^{\epsilon_\text{max}}d\epsilon\left(1 -
 \frac{1}{n^2\beta^2}\right) = \frac{\alpha z^2}{\hbar
 c}\left[\epsilon_\text{max} - \epsilon_\text{min} -
 \frac{1}{\beta^2}\int_{\epsilon_\text{min}}^{\epsilon_\text{max}}
 \frac{d\epsilon}{n^2(\epsilon)}\right].
 * \f]
 */
class CerenkovDndxCalculator
{
  public:
    // Construct from optical properties and Cerenkov angle integrals
    inline CELER_FUNCTION
    CerenkovDndxCalculator(OpticalPropertyRef const& properties,
                           CerenkovRef const& shared,
                           OpticalMaterialId material,
                           real_type charge);

    // Calculate the mean number of Cerenkov photons produced per unit length
    inline CELER_FUNCTION real_type operator()(real_type inv_beta);

  private:
    OpticalPropertyRef const& properties_;
    CerenkovRef const& shared_;
    OpticalMaterialId material_;
    real_type alpha_zsq_over_hbar_c_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from optical properties and Cerenkov angle integrals.
 */
CELER_FUNCTION
CerenkovDndxCalculator::CerenkovDndxCalculator(
    OpticalPropertyRef const& properties,
    CerenkovRef const& shared,
    OpticalMaterialId material,
    real_type charge)
    : properties_(properties)
    , shared_(shared)
    , material_(material)
    , alpha_zsq_over_hbar_c_(1e14 * constants::alpha_fine_structure
                             * constants::e_electron * ipow<2>(charge)
                             / (constants::hbar_planck * constants::c_light))
{
    CELER_EXPECT(properties_);
    CELER_EXPECT(shared_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the mean number of Cerenkov photons produced per unit length.
 */
CELER_FUNCTION real_type CerenkovDndxCalculator::operator()(real_type inv_beta)
{
    CELER_EXPECT(inv_beta > 0);

    if (!shared_.angle_integral[material_])
    {
        // No optical properties for this material
        return 0;
    }

    CELER_ASSERT(material_ < properties_.refractive_index.size());
    GenericCalculator calc_refractive_index(
        properties_.refractive_index[material_], properties_.reals);
    real_type energy_max = calc_refractive_index.grid().back();
    if (inv_beta > calc_refractive_index(energy_max))
    {
        // Incident particle energy is below the threshold for Cerenkov
        // emission (i.e., beta < 1 / n_max)
        return 0;
    }

    GenericCalculator calc_angle_integral(shared_.angle_integral[material_],
                                          shared_.reals);
    real_type angle_integral = calc_angle_integral(energy_max);
    real_type energy_min = calc_refractive_index.grid().front();
    real_type delta_energy;
    if (inv_beta < calc_refractive_index(energy_min))
    {
        delta_energy = energy_max - energy_min;
    }
    else
    {
        // In a dispersive medium the index of refraction is an increasing
        // function of photon energy
        auto grid_data = properties_.refractive_index[material_];
        CELER_ASSERT(is_monotonic_increasing(
            make_span(properties_.reals[grid_data.value])));

        // Find the energy where the refractive index is equal to 1 / beta.
        // Both energy and refractive index are monotonically increasing, so
        // the grid and values can be swapped and the energy can be calculated
        // from a given index of refraction
        trivial_swap(grid_data.grid, grid_data.value);
        auto energy = GenericCalculator(grid_data, properties_.reals)(inv_beta);
        delta_energy = energy_max - energy;
        angle_integral -= calc_angle_integral(energy);
    }

    // Calculate number of photons. This may be negative if the incident
    // particle energy is very close to (just above) the Cerenkov production
    // threshold
    return clamp_to_nonneg(
        alpha_zsq_over_hbar_c_
        * (delta_energy - angle_integral * ipow<2>(inv_beta)));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
