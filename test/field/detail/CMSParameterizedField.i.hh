//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CMSParameterizedField.cc
//---------------------------------------------------------------------------//
#include "CMSParameterizedField.hh"

#include "base/Algorithms.hh"
#include "base/Units.hh"
#include <cmath>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate the magnetic field at the given position inside the CMS tracker.
 * The parameterization is valid only for r < 1.15m and |z| < 2.80m when used
 * with the CMS detector geometry.
 */
CELER_FUNCTION
Real3 CMSParameterizedField::operator()(const Real3& pos)
{
    Real3 value{0., 0., 0.};

    real_type r    = std::sqrt(ipow<2>(pos[0]) + ipow<2>(pos[1]));
    Real3     bw   = this->evaluate_field(r, pos[2]);
    real_type rinv = (r > 0) ? 1 / r : 0;

    value[0] = units::tesla * bw[0] * pos[0] * rinv;
    value[1] = units::tesla * bw[0] * pos[1] * rinv;
    value[2] = units::tesla * bw[2];

    return value;
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the magnetic field value at the given (r, z) position based on
 * the parameterized function.
 *
 * TODO: simplify and optimize
 */
CELER_FUNCTION
Real3 CMSParameterizedField::evaluate_field(real_type r, real_type z)
{
    const real_type prm[9] = {4.24326,
                              15.0201,
                              3.81492,
                              0.0178712,
                              0.000656527,
                              2.45818,
                              0.00778695,
                              2.12500,
                              1.77436};

    real_type ap2   = 4 * ipow<2>(prm[0] / prm[1]);
    real_type hb0   = real_type(0.5) * prm[2] * std::sqrt(1 + ap2);
    real_type hlova = 1 / std::sqrt(ap2);
    real_type ainv  = 2 * hlova / prm[1];
    real_type coeff = 1 / ipow<2>(prm[8]);

    // Convert to m (cms magnetic field parameterization)
    r *= 1 / units::meter;
    z *= 1 / units::meter;
    // The max Bz point is shifted in z
    z -= prm[3];

    real_type az    = std::abs(z);
    real_type zainv = z * ainv;
    real_type u     = hlova - zainv;
    real_type v     = hlova + zainv;

    Real4 fu = this->evaluate_parameters(u);
    Real4 gv = this->evaluate_parameters(v);

    real_type rat  = real_type(0.5) * r * ainv;
    real_type rat2 = ipow<2>(rat);

    Real3 bw;
    bw[0] = hb0 * rat * (fu[1] - gv[1] - (fu[3] - gv[3]) * rat2 * 0.5);
    bw[1] = 0;
    bw[2] = hb0 * (fu[0] + gv[0] - (fu[2] + gv[2]) * rat2);
    real_type corBr = prm[4] * r * z * ipow<2>(az - prm[5]);
    real_type corBz = -prm[6]
                      * (std::exp(-ipow<2>(z - prm[7]) * coeff)
                         + std::exp(-ipow<2>(z + prm[7]) * coeff));
    bw[0] += corBr;
    bw[2] += corBz;

    return bw;
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the parameterization function and its 3 derivatives.
 */
CELER_FUNCTION
CMSParameterizedField::Real4
CMSParameterizedField::evaluate_parameters(real_type x)
{
    real_type a = 1 / (1 + ipow<2>(x));
    real_type b = std::sqrt(a);

    Real4 ff;
    ff[0] = x * b;
    ff[1] = a * b;
    ff[2] = -3 * x * a * ff[1];
    ff[3] = a * ff[2] * ((1 / x) - 4 * x);

    return ff;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
