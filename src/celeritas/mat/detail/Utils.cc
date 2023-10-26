//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/detail/Utils.cc
//---------------------------------------------------------------------------//
#include "Utils.hh"

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Constants.hh"

#include "../MaterialData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate Coulomb correction factor (unitless).
 *
 * This uses the formulation of arXiv:1204.3675 [hep-ph] of the correction
 * factor presented in section 34.4.1 of the Review of Particle Physics (2020).
 * The calculation of the values is in the celeritas-docs repo at
 * https://github.com/celeritas-project/celeritas-docs/blob/master/notes/coulomb-correction/eval.py
 *
 * This is accurate to 16 digits of precision through Z=92, as compared to 5
 * or 6 digits for equation 34.26 which appears in RPP.
 */
real_type calc_coulomb_correction(AtomicNumber atomic_number)
{
    CELER_EXPECT(atomic_number);
    using constants::alpha_fine_structure;

    static double const zeta[] = {2.0205690315959429e-01,
                                  3.6927755143369927e-02,
                                  8.3492773819228271e-03,
                                  2.0083928260822143e-03,
                                  4.9418860411946453e-04,
                                  1.2271334757848915e-04,
                                  3.0588236307020493e-05,
                                  7.6371976378997626e-06,
                                  1.9082127165539390e-06,
                                  4.7693298678780645e-07,
                                  1.1921992596531106e-07,
                                  2.9803503514652279e-08,
                                  7.4507117898354301e-09,
                                  1.8626597235130491e-09,
                                  4.6566290650337837e-10,
                                  1.1641550172700519e-10};

    const real_type alphazsq
        = ipow<2>(alpha_fine_structure * atomic_number.unchecked_get());
    real_type fz = 1 / (1 + alphazsq);
    real_type azpow = 1;
    for (double zeta_i : zeta)
    {
        fz += azpow * zeta_i;
        azpow *= -alphazsq;
    }

    CELER_ENSURE(fz > 0);
    return alphazsq * fz;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate inverse of Tsai radiation length for bremsstrahlung [cm^2/g].
 *
 * See ElementView::mass_radiation_coeff for details on this calculation and
 * how it's used.
 */
real_type calc_mass_rad_coeff(ElementRecord const& el)
{
    CELER_EXPECT(el.atomic_number);
    CELER_EXPECT(el.atomic_mass > zero_quantity());
    CELER_EXPECT(el.coulomb_correction > 0);
    using constants::alpha_fine_structure;
    using constants::r_electron;

    const real_type z_real = el.atomic_number.unchecked_get();

    // Table 34.2 for calculating lrad/lrad prime
    real_type lrad, lrad_prime;
    switch (el.atomic_number.unchecked_get())
    {
        // clang-format off
        case 1: lrad = 5.31; lrad_prime = 6.144; break;
        case 2: lrad = 4.79; lrad_prime = 5.621; break;
        case 3: lrad = 4.74; lrad_prime = 5.805; break;
        case 4: lrad = 4.71; lrad_prime = 5.924; break;
            // clang-format on
        default:
            lrad = std::log(184.15 * std::pow(z_real, real_type(-1) / 3));
            lrad_prime = std::log(1194.0 * std::pow(z_real, real_type(-2) / 3));
    }

    // Eq 34.25
    constexpr real_type inv_x0_factor = 4 * alpha_fine_structure
                                        * ipow<2>(r_electron);
    return inv_x0_factor / native_value_from(el.atomic_mass)
           * (ipow<2>(z_real) * (lrad - el.coulomb_correction)
              + z_real * lrad_prime);
}

//---------------------------------------------------------------------------//
/*!
 * Get the mean excitation energy of an element (MeV).
 *
 * The mean excitation energy for all elements are the ICRU recommended values
 * from "Stopping powers for electrons and positrons", ICRU Report 37 (1984).
 *
 * TODO: in Geant4, the mean excitation energies for many compounds are stored
 * rather than calculated as the average over elements.
 */
units::MevEnergy get_mean_excitation_energy(AtomicNumber atomic_number)
{
    CELER_EXPECT(atomic_number);
    // Mean excitation energy for Z=1-98 [eV]
    static double const mean_excitation_energy[] = {
        19.2,  41.8,  40.0,  63.7,  76.0,  81.0,  82.0,  95.0,  115.0, 137.0,
        149.0, 156.0, 166.0, 173.0, 173.0, 180.0, 174.0, 188.0, 190.0, 191.0,
        216.0, 233.0, 245.0, 257.0, 272.0, 286.0, 297.0, 311.0, 322.0, 330.0,
        334.0, 350.0, 347.0, 348.0, 343.0, 352.0, 363.0, 366.0, 379.0, 393.0,
        417.0, 424.0, 428.0, 441.0, 449.0, 470.0, 470.0, 469.0, 488.0, 488.0,
        487.0, 485.0, 491.0, 482.0, 488.0, 491.0, 501.0, 523.0, 535.0, 546.0,
        560.0, 574.0, 580.0, 591.0, 614.0, 628.0, 650.0, 658.0, 674.0, 684.0,
        694.0, 705.0, 718.0, 727.0, 736.0, 746.0, 757.0, 790.0, 790.0, 800.0,
        810.0, 823.0, 823.0, 830.0, 825.0, 794.0, 827.0, 826.0, 841.0, 847.0,
        878.0, 890.0, 902.0, 921.0, 934.0, 939.0, 952.0, 966.0};

    int idx = atomic_number.unchecked_get() - 1;
    CELER_ASSERT(idx * sizeof(double) < sizeof(mean_excitation_energy));
    return units::MevEnergy(1e-6 * mean_excitation_energy[idx]);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
