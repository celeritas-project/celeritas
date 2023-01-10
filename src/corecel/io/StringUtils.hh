//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StringUtils.hh
//! \brief Helper functions for string processing
//---------------------------------------------------------------------------//
#pragma once

#include <string>

namespace celeritas
{
//---------------------------------------------------------------------------//
// Whether the string starts with another string.
bool starts_with(std::string const& main_string, std::string const& prefix);

//---------------------------------------------------------------------------//
// Whether the string ends with another string.
bool ends_with(std::string const& main_string, std::string const& suffix);

//---------------------------------------------------------------------------//
}  // namespace celeritas
