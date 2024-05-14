//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/GeneralQuadric.cc
//---------------------------------------------------------------------------//
#include "GeneralQuadric.hh"

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/math/Algorithms.hh"

#include "SimpleQuadric.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with all coefficients.
 *
 * We normalize the coefficients so the infinity-norm of the terms is unity.
 *
 * \todo Use a more rigorous method to normalize
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
    static constexpr auto size = StorageSpan::extent;
    real_type norm{0};
    for (auto v : Span<real_type, size>{&a_, size})
    {
        norm = fmax(std::fabs(v), norm);
    }
    CELER_VALIDATE(norm != 0,
                   << "quadric coefficients are all zeros (degenerate)");
    norm = 1 / norm;
    for (real_type& v : Span<real_type, size>{&a_, size})
    {
        v *= norm;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Promote from a simple quadric.
 */
GeneralQuadric::GeneralQuadric(SimpleQuadric const& other) noexcept
    : GeneralQuadric{make_array(other.second()),
                     Real3{0, 0, 0},
                     make_array(other.first()),
                     other.zeroth()}
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
