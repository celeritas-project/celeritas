//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/GeneralQuadric.cc
//---------------------------------------------------------------------------//
#include "GeneralQuadric.hh"

#include "corecel/Assert.hh"

#include "SimpleQuadric.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with all coefficients.
 *
 * Note that no normalization takes place: the user-provided scaling is
 * preserved to prevent false elimination of second-order terms in surface
 * deduplication.
 */
GeneralQuadric::GeneralQuadric(Real3 const& abc,
                               Real3 const& def,
                               Real3 const& ghi,
                               real_type j)
    : a_(abc[0])
    , b_(abc[1])
    , c_(abc[2])
    , d_(def[0])
    , e_(def[1])
    , f_(def[2])
    , g_(ghi[0])
    , h_(ghi[1])
    , i_(ghi[2])
    , j_(j)
{
    CELER_EXPECT(a_ != 0 || b_ != 0 || c_ != 0 || d_ != 0 || e_ != 0 || f_ != 0
                 || g_ != 0 || h_ != 0 || i_ != 0);
}

//---------------------------------------------------------------------------//
/*!
 * Promote from a simple quadric.
 */
GeneralQuadric::GeneralQuadric(SimpleQuadric const& other) noexcept(
    !CELERITAS_DEBUG)
    : GeneralQuadric{make_array(other.second()),
                     Real3{0, 0, 0},
                     make_array(other.first()),
                     other.zeroth()}
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
