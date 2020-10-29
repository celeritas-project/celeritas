//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ColorUtils.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
// Whether colors are enabled (currently read-only)
bool use_color();

//---------------------------------------------------------------------------//
// Get an ANSI color code: [y]ellow / [r]ed / [ ]default / ...
const char* color_code(char abbrev);

//---------------------------------------------------------------------------//
} // namespace celeritas
