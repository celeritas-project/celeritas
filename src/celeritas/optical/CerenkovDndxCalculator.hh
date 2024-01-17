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
#include "celeritas/Quantities.hh"
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
   \dif N = \frac{\alpha z^2}{\hbar c}\sin^2\theta \dif\epsilon \dif x =
   \frac{\alpha z^2}{\hbar c}\left(1 - \frac{1}{n^2\beta^2}\right) \dif\epsilon
   \dif x,
 * \f]
 * where \f$ n \f$ is the refractive index of the material, \f$ \epsilon \f$
 * is the photon energy, and \f$ \theta \f$ is the angle of the emitted photons
 * with respect to the incident particle direction, given by \f$ \cos\theta = 1
 * / (\beta n) \f$. Note that in a dispersive medium, the index of refraction
 * is an inreasing function of photon energy. The mean number of photons per
 * unit length is given by
 * \f[
   \dif N/\dif x = \frac{\alpha z^2}{\hbar c}
   \int_{\epsilon_\text{min}}^{\epsilon_\text{max}} \dif\epsilon \left(1 -
   \frac{1}{n^2\beta^2} \right) = \frac{\alpha z^2}{\hbar c}
   \left[\epsilon_\text{max} - \epsilon_\text{min} - \frac{1}{\beta^2}
   \int_{\epsilon_\text{min}}^{\epsilon_\text{max}}
   \frac{\dif\epsilon}{n^2(\epsilon)} \right].
 * \f]
 */
class CerenkovDndxCalculator
{
  public:
    // Construct from optical properties and Cerenkov angle integrals
    inline CELER_FUNCTION
    CerenkovDndxCalculator(NativeCRef<OpticalPropertyData> const& properties,
                           NativeCRef<CerenkovData> const& shared,
                           OpticalMaterialId material,
                           units::ElementaryCharge charge);

    // Calculate the mean number of Cerenkov photons produced per unit length
    inline CELER_FUNCTION real_type operator()(units::LightSpeed beta);

  private:
    NativeCRef<OpticalPropertyData> const& properties_;
    NativeCRef<CerenkovData> const& shared_;
    OpticalMaterialId material_;
    real_type zsq_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from optical properties and Cerenkov angle integrals.
 */
CELER_FUNCTION
CerenkovDndxCalculator::CerenkovDndxCalculator(
    NativeCRef<OpticalPropertyData> const& properties,
    NativeCRef<CerenkovData> const& shared,
    OpticalMaterialId material,
    units::ElementaryCharge charge)
    : properties_(properties)
    , shared_(shared)
    , material_(material)
    , zsq_(ipow<2>(charge.value()))
{
    CELER_EXPECT(properties_);
    CELER_EXPECT(shared_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the mean number of Cerenkov photons produced per unit length.
 */
CELER_FUNCTION real_type
CerenkovDndxCalculator::operator()(units::LightSpeed beta)
{
    CELER_EXPECT(beta.value() > 0 && beta.value() <= 1);

    if (!shared_.angle_integral[material_])
    {
        // No optical properties for this material
        return 0;
    }

    CELER_ASSERT(material_ < properties_.refractive_index.size());
    real_type inv_beta = 1 / beta.value();
    GenericCalculator calc_refractive_index(
        properties_.refractive_index[material_], properties_.reals);
    real_type energy_max = calc_refractive_index.grid().back();
    if (inv_beta > calc_refractive_index(energy_max))
    {
        // Incident particle energy is below the threshold for Cerenkov
        // emission (i.e., beta < 1 / n_max)
        return 0;
    }

    // Calculate the Cerenkov angle integral [MeV]
    GenericCalculator calc_integral(shared_.angle_integral[material_],
                                    shared_.reals);

    // Calculate \f$ \int_{\epsilon_\text{min}}^{\epsilon_\text{max}}
    // \dif\epsilon \left(1 - \frac{1}{n^2\beta^2}\right) \f$
    real_type energy;
    if (inv_beta < calc_refractive_index[0])
    {
        energy = energy_max - calc_refractive_index.grid().front()
                 - calc_integral(energy_max) * ipow<2>(inv_beta);
    }
    else
    {
        // TODO: Check that refractive index is monotonically increasing when
        // grids are imported

        // Find the energy where the refractive index is equal to 1 / beta.
        // Both energy and refractive index are monotonically increasing, so
        // the grid and values can be swapped and the energy can be calculated
        // from a given index of refraction
        auto grid_data = properties_.refractive_index[material_];
        trivial_swap(grid_data.grid, grid_data.value);
        real_type energy_min
            = GenericCalculator(grid_data, properties_.reals)(inv_beta);
        energy = energy_max - energy_min
                 - (calc_integral(energy_max) - calc_integral(energy_min))
                       * ipow<2>(inv_beta);
    }

    // Calculate number of photons. This may be negative if the incident
    // particle energy is very close to (just above) the Cerenkov production
    // threshold
    return clamp_to_nonneg(zsq_ * constants::alpha_fine_structure
                           / (constants::hbar_planck * constants::c_light)
                           * native_value_from(units::MevEnergy(energy)));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
