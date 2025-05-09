//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CherenkovDndxCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/grid/VectorUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"

#include "CherenkovData.hh"
#include "MaterialView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the mean number of Cherenkov photons produced per unit length.
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
   \difd{N}{x} = \frac{\alpha z^2}{\hbar c}
   \int_{\epsilon_\text{min}}^{\epsilon_\text{max}} \left(1 -
   \frac{1}{n^2\beta^2} \right) \dif\epsilon
   = \frac{\alpha z^2}{\hbar c}
   \left[\epsilon_\text{max} - \epsilon_\text{min} - \frac{1}{\beta^2}
   \int_{\epsilon_\text{min}}^{\epsilon_\text{max}}
   \frac{1}{n^2(\epsilon)}\dif\epsilon \right].
 * \f]
 */
class CherenkovDndxCalculator
{
  public:
    // Construct from optical materials and Cherenkov angle integrals
    inline CELER_FUNCTION
    CherenkovDndxCalculator(MaterialView const& material,
                            NativeCRef<CherenkovData> const& shared,
                            units::ElementaryCharge charge);

    // Calculate the mean number of Cherenkov photons produced per unit length
    inline CELER_FUNCTION real_type operator()(units::LightSpeed beta);

  private:
    // Calculate refractive index [MeV -> unitless]
    NonuniformGridCalculator calc_refractive_index_;

    // Calculate the Cherenkov angle integral [MeV -> unitless]
    NonuniformGridCalculator calc_integral_;

    // Square of particle charge
    real_type zsq_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from optical materials and Cherenkov angle integrals.
 */
CELER_FUNCTION
CherenkovDndxCalculator::CherenkovDndxCalculator(
    MaterialView const& material,
    NativeCRef<CherenkovData> const& shared,
    units::ElementaryCharge charge)
    : calc_refractive_index_(material.make_refractive_index_calculator())
    , calc_integral_{shared.angle_integral[material.material_id()],
                     shared.reals}
    , zsq_(ipow<2>(charge.value()))
{
    CELER_EXPECT(charge != zero_quantity());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the mean number of Cherenkov photons produced per unit length.
 *
 * \todo define a "generic grid extents" class for finding the lower/upper x/y
 * coordinates on grid data. In the future we could cache these if the memory
 * lookups result in too much indirection.
 */
CELER_FUNCTION real_type
CherenkovDndxCalculator::operator()(units::LightSpeed beta)
{
    CELER_EXPECT(beta.value() > 0 && beta.value() <= 1);

    real_type inv_beta = 1 / beta.value();
    real_type energy_max = calc_refractive_index_.grid().back();
    if (inv_beta > calc_refractive_index_(energy_max))
    {
        // Incident particle energy is below the threshold for Cherenkov
        // emission (i.e., beta < 1 / n_max)
        return 0;
    }

    // Calculate \f$ \int_{\epsilon_\text{min}}^{\epsilon_\text{max}}
    // \dif\epsilon \left(1 - \frac{1}{n^2\beta^2}\right) \f$
    real_type energy;
    if (inv_beta < calc_refractive_index_[0])
    {
        energy = energy_max - calc_refractive_index_.grid().front()
                 - calc_integral_(energy_max) * ipow<2>(inv_beta);
    }
    else
    {
        // Find the energy where the refractive index is equal to 1 / beta.
        // Both energy and refractive index are monotonically increasing, so
        // the grid and values can be swapped and the energy can be calculated
        // from a given index of refraction
        real_type energy_min = calc_refractive_index_.make_inverse()(inv_beta);
        energy = energy_max - energy_min
                 - (calc_integral_(energy_max) - calc_integral_(energy_min))
                       * ipow<2>(inv_beta);
    }

    // Calculate number of photons. This may be negative if the incident
    // particle energy is very close to (just above) the Cherenkov production
    // threshold
    return clamp_to_nonneg(zsq_
                           * (constants::alpha_fine_structure
                              / (constants::hbar_planck * constants::c_light))
                           * native_value_from(units::MevEnergy(energy)));
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
