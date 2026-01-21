//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ColorUtils.hh
//! \brief Helper functions for writing colors to the terminal
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Whether colors are enabled (currently read-only)
bool use_color();

//---------------------------------------------------------------------------//
// Get an ANSI color code: [y]ellow / [r]ed / [\0]reset / ...
char const* ansi_color(char abbrev = '\0');

//---------------------------------------------------------------------------//
// DEPRECATED (remove in v1.0): get an ANSI color code
CELER_FORCEINLINE char const* color_code(char abbrev)
{
    return ansi_color(abbrev);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
