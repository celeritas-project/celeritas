//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InteractionIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include "physics/base/Interaction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Write a host-side Interaction to a stream for debugging.
std::ostream& operator<<(std::ostream& os, const Interaction& i);

//---------------------------------------------------------------------------//
} // namespace celeritas
