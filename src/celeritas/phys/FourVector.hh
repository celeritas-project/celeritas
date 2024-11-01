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
#include "corecel/math/ArrayUtils.hh"
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
    Real3 mom{0, 0, 0};  //!< Particle momentum
    real_type energy{0};  //!< Particle total energy (\f$\sqrt{p^2 + m^2}\f$)

    // Assignment operator (+=)
    inline CELER_FUNCTION FourVector& operator+=(FourVector const& v)
    {
        mom += v.mom;
        energy += v.energy;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// INLINE UTILITY FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Add two four-vectors.
 */
inline CELER_FUNCTION FourVector operator+(FourVector const& lhs,
                                           FourVector const& rhs)
{
    FourVector result = lhs;
    return result += rhs;
}

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
    real_type const v_sq = dot_product(v, v);
    CELER_EXPECT(v_sq < real_type{1});

    real_type const vp = dot_product(v, p->mom);
    real_type const gamma = real_type{1} / std::sqrt(1 - v_sq);
    real_type const lambda = (v_sq > 0 ? (gamma - 1) * vp / v_sq : 0)
                             + gamma * p->energy;

    axpy(lambda, v, &(p->mom));
    p->energy = gamma * (p->energy + vp);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the magnitude of a four vector.
 */
inline CELER_FUNCTION real_type norm(FourVector const& a)
{
    return std::sqrt(std::fabs(ipow<2>(a.energy) - dot_product(a.mom, a.mom)));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
