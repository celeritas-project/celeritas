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
{
    CELER_EXPECT(size > 0);
    resize(&device_, size, mats.max_element_components());
    device_ref_ = device_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
