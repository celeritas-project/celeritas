//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//---------------------------------------------------------------------------//
#ifndef base_Types_hh
#define base_Types_hh

namespace celeritas
{
//---------------------------------------------------------------------------//
template<typename T, std::size_t N>
class array;

using ssize_type   = int;
using real_type    = double;
using RealPointer3 = array<real_type*, 3>;
using Real3        = array<real_type, 3>;

//---------------------------------------------------------------------------//

enum class Interp
{
    Linear,
    Log
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // base_Types_hh
