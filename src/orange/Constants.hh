//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/Constants.hh
//! \todo move to corecel/
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

namespace celeritas
{
namespace constants
{
//---------------------------------------------------------------------------//

#define CELER_ICRT_ inline constexpr real_type

//!@{
//! \name Mathemetical constants (truncated)
CELER_ICRT_ pi = 3.14159265358979323846;
CELER_ICRT_ euler = 2.71828182845904523536;
CELER_ICRT_ sqrt_two = 1.41421356237309504880;
CELER_ICRT_ sqrt_three = 1.73205080756887729353;
//!@}

#undef CELER_ICRT_

//---------------------------------------------------------------------------//
}  // namespace constants
}  // namespace celeritas
