//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//---------------------------------------------------------------------------//
#ifndef base_Types_hh
#define base_Types_hh

#include <cstddef>
#include "OpaqueId.hh"

namespace celeritas
{
template<typename T, std::size_t N>
class array;
template<typename T, std::size_t N>
class span;

class Thread;
//---------------------------------------------------------------------------//
using size_type    = std::size_t;
using ssize_type   = int;
using real_type    = double;
using RealPointer3 = array<real_type*, 3>;
using Real3        = array<real_type, 3>;
using SpanReal3    = span<real_type, 3>;

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

#endif // base_Types_hh
