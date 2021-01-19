//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialStateStore.cc
//---------------------------------------------------------------------------//
#include "MaterialStateStore.hh"

#include "MaterialParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from material params and number of track states.
 */
MaterialStateStore::MaterialStateStore(const MaterialParams& mats,
                                       size_type             size)
    : max_el_(mats.max_element_components())
    , states_(size)
    , element_scratch_(size * max_el_)
{
    CELER_EXPECT(size > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Get the interface to on-device states.
 */
MaterialStatePointers MaterialStateStore::device_pointers()
{
    MaterialStatePointers result;
    result.state           = states_.device_pointers();
    result.element_scratch = element_scratch_.device_pointers();

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
