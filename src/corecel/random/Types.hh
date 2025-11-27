//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// ENUMERATIONS
//---------------------------------------------------------------------------//

//! Distributions returning a single real type
enum class OnedDistributionType
{
    delta,
    normal,
    size_
};

//! Distributions returning a length three array of real types
enum class ThreedDistributionType
{
    delta,
    isotropic,
    uniform_box,
    size_
};

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Identifier for a distribution returning a single real type
using OnedDistributionId = OpaqueId<OnedDistributionType>;

//! Identifier for a distribution returning a length three array
using ThreedDistributionId = OpaqueId<ThreedDistributionType>;

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

char const* to_cstring(OnedDistributionType);
char const* to_cstring(ThreedDistributionType);

//---------------------------------------------------------------------------//
}  // namespace celeritas
