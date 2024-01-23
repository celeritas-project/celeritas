//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/InteractionIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>

#include "celeritas/phys/Interaction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Write a host-side Interaction to a stream for debugging.
std::ostream& operator<<(std::ostream& os, Interaction const& i);

//---------------------------------------------------------------------------//
}  // namespace celeritas
