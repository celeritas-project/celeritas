//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/XorwowRngParams.cc
//---------------------------------------------------------------------------//
#include "XorwowRngParams.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Array.hh"
#include "celeritas/random/XorwowRngData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a low-entropy seed.
 */
XorwowRngParams::XorwowRngParams(unsigned int seed)
{
    HostVal<XorwowRngParamsData> host_data;
    host_data.seed = {seed};
    CELER_ASSERT(host_data);
    data_ = CollectionMirror<XorwowRngParamsData>{std::move(host_data)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
