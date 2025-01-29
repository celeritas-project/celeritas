//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/FourVector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/Types.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// STRUCTS
//---------------------------------------------------------------------------//
/*!
 * The momentum-energy four-vector (Lorentz vector).
 *
 * The units of this class are implicit: momentum is \c MevMomentum
 * and energy is \c MevEnergy.
 */
struct FourVector
{
    using Energy = units::MevEnergy;
    using Momentum = units::MevMomentum;
    using Mass = units::MevMass;

    Real3 mom{0, 0, 0};  //!< Particle momentum
    real_type energy{0};  //!< Particle total energy (\f$\sqrt{p^2 + m^2}\f$)

    // Construct from a particle and direction
    template<class PTV>
    static inline CELER_FUNCTION FourVector
    from_particle(PTV const& particle, Real3 const& direction);

    // Construct from momentum, rest mass, direction
    static inline CELER_FUNCTION FourVector
    from_mass_momentum(Mass m, Momentum p, Real3 const& direction);

    //! In-place addition
    CELER_FUNCTION FourVector& operator+=(FourVector const& v)
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
 * Construct from rest mass, momentum, direction.
 *
 * Note that this could be improved by using \c std::hypot which yields more
 * accurate answers if the magnitudes of the momentum and mass are very
 * different. However, differences in the implementation of that function can
 * lead to differences across platforms, compilers, and architectures, so for
 * now we use a naive sqrt+ipow.
 */
CELER_FUNCTION FourVector FourVector::from_mass_momentum(Mass m,
                                                         Momentum p,
                                                         Real3 const& direction)
{
    return {p.value() * direction,
            std::sqrt(ipow<2>(p.value()) + ipow<2>(m.value()))};
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a particle and direction.
 */
template<class PTV>
CELER_FUNCTION FourVector FourVector::from_particle(PTV const& particle,
                                                    Real3 const& direction)
{
    return {direction * value_as<Momentum>(particle.momentum()),
            value_as<Energy>(particle.total_energy())};
}

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
 * Perform the Lorentz transformation along a boost vector.
 *
 * The transformation (\f$ \Lambda^{\alpha}_{\beta} \f$) along
 * the boost vector (\f$ \vec{v} \f$) for a four-vector \f$ p^{\beta} \f$ is:
 *
 * \f[ p^{\prime \beta} = \Lambda^{\alpha}_{\beta} (\vec{v}) p^{\beta} \f].
 *
 * \todo Define a boost function that takes a second FourVector for reduced
 * register usage
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
