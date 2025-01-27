//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/SecondaryIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>

#include "celeritas/phys/Secondary.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Write a host-side Secondary to a stream for debugging.
std::ostream& operator<<(std::ostream& os, Secondary const& s);

//---------------------------------------------------------------------------//
}  // namespace celeritas
