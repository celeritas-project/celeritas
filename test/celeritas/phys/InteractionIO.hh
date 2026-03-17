//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/InteractionIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>

#include "corecel/io/StreamUtils.hh"  // IWYU pragma: export
#include "celeritas/phys/Interaction.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Write a host-side Interaction to a stream for debugging.
std::ostream& operator<<(std::ostream& os, Interaction const& i);

//! Allow printing of pos/dir for convenience
inline std::string to_string(Real3 const& r)
{
    return stream_to_string(r);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
