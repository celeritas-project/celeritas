//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/FourVector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Array.hh"
#include "corecel/math/ArrayOperators.hh"
#include "geocel/Types.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// STRUCTS
//---------------------------------------------------------------------------//
/*!
 * The momentum-energy four-vector (Lorentz vector).
 */
struct FourVector
{
    //// DATA ////

    Real3 mom{0, 0, 0};  //!< Particle momentum
    real_type energy{0};  //!< Particle energy
};

//---------------------------------------------------------------------------//
// INLINE UTILITY FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get the boost vector (\f$ \frac{\vec{mom}}/{energy} \f$) of a four-vector.
 */
inline CELER_FUNCTION Real3 boost_vector(FourVector const& p)
{
    CELER_EXPECT(p.energy > 0);
    return (real_type{1} / p.energy) * p.mom;
}

//---------------------------------------------------------------------------//
/*!
 * Perform the Lorentz transformation (\f$ \Lambda^{\alpha}_{\beta} \f$) along
 * the boost vector (\f$ \vec{v} \f$) for a four-vector \f$ p^{\beta} \f$:
 *
 * \f$ p^{\prime \beta} = \Lambda^{\alpha}_{\beta} (\vec{v}) p^{\beta} \f$.
 *
 */
inline CELER_FUNCTION void boost(Real3 const& v, FourVector* p)
{
    const real_type v_sq = dot_product(v, v);
    CELER_EXPECT(v_sq < real_type{1});

    const real_type vp = dot_product(v, p->mom);
    const real_type gamma = real_type{1} / std::sqrt(1 - v_sq);
    const real_type lambda = (v_sq > 0 ? (gamma - 1) * vp / v_sq : 0)
                             + gamma * p->energy;

    axpy(lambda, v, &(p->mom));
    p->energy = gamma * (p->energy + vp);
}

}  // namespace celeritas
