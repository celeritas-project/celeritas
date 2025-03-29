//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/Constants.hh
//! \brief Mathematical constants. See \c celeritas/Constants.hh for physical.
//---------------------------------------------------------------------------//
#pragma once

#include "math/Constant.hh"

namespace celeritas
{
namespace constants
{
//---------------------------------------------------------------------------//

#define CELER_ICRT_ inline constexpr Constant

//!@{
//! \name Mathemetical constants (truncated)
CELER_ICRT_ pi{3.14159265358979323846};
CELER_ICRT_ sqrt_pi{1.77245385090551602730};

CELER_ICRT_ euler{2.71828182845904523536};
CELER_ICRT_ sqrt_euler{1.64872127070012814685};

CELER_ICRT_ sqrt_two{1.41421356237309504880};
CELER_ICRT_ sqrt_three{1.73205080756887729353};
//!@}

#undef CELER_ICRT_

//---------------------------------------------------------------------------//
}  // namespace constants
}  // namespace celeritas
