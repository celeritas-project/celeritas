//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/GeneralQuadric.cc
//---------------------------------------------------------------------------//
#include "GeneralQuadric.hh"

#include "SimpleQuadric.hh"

namespace celeritas
{
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
