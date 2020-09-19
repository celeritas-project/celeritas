//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include "Array.hh"
#include "OpaqueId.hh"

namespace celeritas
{
struct Thread;
//---------------------------------------------------------------------------//
using size_type    = std::size_t;
using real_type    = double;
using RealPointer3 = array<real_type*, 3>;
using Real3        = array<real_type, 3>;

//! Index of the current CUDA thread, with type safety for containers.
using ThreadId = OpaqueId<Thread, unsigned int>;

//---------------------------------------------------------------------------//

enum class Interp
{
    Linear,
    Log
};

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
