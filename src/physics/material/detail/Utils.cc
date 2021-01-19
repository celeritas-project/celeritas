//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.cc
//---------------------------------------------------------------------------//
#include "Utils.hh"

#include <algorithm>
#include <cmath>
#include "base/Algorithms.hh"
#include "base/Constants.hh"
#include "base/Range.hh"
#include "base/Quantity.hh"
#include "physics/material/ElementDef.hh"

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
real_type calc_coulomb_correction(int atomic_number)
{
    CELER_EXPECT(atomic_number > 0);
    using constants::alpha_fine_structure;

    const real_type alphazsq = ipow<2>(alpha_fine_structure * atomic_number);

    static const double zeta[] = {2.0205690315959429e-01,
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
    real_type           fz     = 1 / (1 + alphazsq);
    real_type           azpow  = 1;
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
real_type calc_mass_rad_coeff(const ElementDef& el)
{
    CELER_EXPECT(el.atomic_number > 0);
    CELER_EXPECT(el.atomic_mass > zero_quantity());
    CELER_EXPECT(el.coulomb_correction > 0);
    using constants::alpha_fine_structure;
    using constants::re_electron;

    const real_type z_real = el.atomic_number;

    // Table 34.2 for calculating lrad/lrad prime
    real_type lrad, lrad_prime;
    switch (el.atomic_number)
    {
        // clang-format off
        case 1: lrad = 5.31; lrad_prime = 6.144; break;
        case 2: lrad = 4.79; lrad_prime = 5.621; break;
        case 3: lrad = 4.74; lrad_prime = 5.805; break;
        case 4: lrad = 4.71; lrad_prime = 5.924; break;
            // clang-format on
        default:
            lrad       = std::log(184.15 * std::pow(z_real, -1.0 / 3));
            lrad_prime = std::log(1194.0 * std::pow(z_real, -2.0 / 3));
    }

    // Eq 34.25
    constexpr real_type inv_x0_factor = 4 * alpha_fine_structure * re_electron
                                        * re_electron;
    return inv_x0_factor / unit_cast(el.atomic_mass)
           * (z_real * z_real * (lrad - el.coulomb_correction)
              + z_real * lrad_prime);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
