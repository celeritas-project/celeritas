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
#include "base/Constants.hh"
#include "base/Quantity.hh"
#include "physics/material/ElementDef.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate inverse of Tsai radiation length for bremsstrahlung [cm^2/g].
 *
 * See \S33.4.2, pp.452--453, from Review of Particle Physics, calculation of
 * 1/X0. This is a
 * semi-empirical formula. The result is like an inverse mass attenuation
 * coefficient: it must be multiplied by density to produce a length.
 */
real_type calc_mass_rad_coeff(const ElementDef& el)
{
    REQUIRE(el.atomic_number > 0);
    REQUIRE(el.atomic_mass > zero_quantity());
    using constants::alpha_fine_structure;
    using constants::re_electron;

    const real_type z_real = el.atomic_number;

    // Table 33.2 for calculating lrad/lrad prime
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

    // Eq 33.27 for calculating f(Z), the coulomb correction factor
    real_type alphazsq = alpha_fine_structure * std::min(el.atomic_number, 92);
    alphazsq *= alphazsq;
    const real_type fz = alphazsq
                         * (1 / (1 + alphazsq) + 0.20206 - 0.0369 * alphazsq
                            + 0.0083 * alphazsq * alphazsq
                            - 0.002 * alphazsq * alphazsq * alphazsq);

    // Eq 33.26
    constexpr real_type inv_x0_factor = 4 * alpha_fine_structure * re_electron
                                        * re_electron;
    return inv_x0_factor / unit_cast(el.atomic_mass)
           * (z_real * z_real * (lrad - fz) + z_real * lrad_prime);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
