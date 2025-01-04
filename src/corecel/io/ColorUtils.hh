//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ColorUtils.hh
//! \brief Helper functions for writing colors to the terminal
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
// Whether colors are enabled (currently read-only)
bool use_color();

//---------------------------------------------------------------------------//
// Get an ANSI color code: [y]ellow / [r]ed / [ ]default / ...
char const* color_code(char abbrev);

//---------------------------------------------------------------------------//
}  // namespace celeritas
