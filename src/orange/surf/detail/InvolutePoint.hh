//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/InvolutePoint.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "orange/OrangeTypes.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Find point on involute using involute equation.
 *
 *  * \f[
 *   x = r_b (\cos(t+a) + t \sin(t+a))
 * \f]
 * \f[
 *   y = r_b (\sin(t+a) - t \cos(t+a))
 * \f]
 *
 */
class InvolutePoint
{
  public:
    //!@{
    //! \name Type aliases
    using Real2 = Array<real_type, 2>;
    //!@}

  public:
    // Construct involute from parameters
    inline CELER_FUNCTION InvolutePoint(real_type r_b, real_type a);

    // Calculate point on an involute
    inline CELER_FUNCTION Real2 operator()(real_type theta) const;

    //// ACCESSORS ////

    //! Get involute parameters
    CELER_FUNCTION real_type r_b() const { return r_b_; }
    CELER_FUNCTION real_type a() const { return a_; }

  private:
    //// DATA ////

    // Involute parameters
    real_type r_b_;
    real_type a_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from involute parameters.
 */
CELER_FUNCTION InvolutePoint::InvolutePoint(real_type r_b, real_type a)
    : r_b_(r_b), a_(a)
{
    CELER_EXPECT(r_b > 0);
    CELER_EXPECT(a >= 0);
}

/*!
 * Calculate the point on an involute.
 */
CELER_FUNCTION Real2 InvolutePoint::operator()(real_type theta) const
{
    real_type angle = theta + a_;
    Real2 point;
    // TODO: check that compiler avoids recomputing trig functions
    point[0] = r_b_ * (std::cos(angle) + theta * std::sin(angle));
    point[1] = r_b_ * (std::sin(angle) - theta * std::cos(angle));

    return point;
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
