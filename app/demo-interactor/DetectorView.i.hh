//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DetectorView.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
CELER_FUNCTION DetectorView::DetectorView(const DetectorPointers& pointers)
    : allocate_(pointers.hit_buffer)
{
}

//---------------------------------------------------------------------------//
/*!
 * Push the given hit onto the back of the detector stack.
 */
CELER_FUNCTION void DetectorView::operator()(const Hit& hit)
{
    REQUIRE(hit.thread);
    REQUIRE(hit.time > 0);
    REQUIRE(hit.energy_deposited > zero_quantity());

    // Allocate and assign the given hit
    Hit* allocated = this->allocate_(1);
    CHECK(allocated);
    *allocated = hit;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
