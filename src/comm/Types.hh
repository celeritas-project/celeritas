//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.hh
//---------------------------------------------------------------------------//
#ifndef comm_Types_hh
#define comm_Types_hh

#include "celeritas_config.h"
#if CELERITAS_USE_MPI
#    include <mpi.h>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
#if CELERITAS_USE_MPI
using MpiComm = MPI_Comm;
#else

struct MpiComm
{
    int value_;
};

inline bool operator==(MpiComm a, MpiComm b)
{
    return a.value_ == b.value_;
}

inline bool operator!=(MpiComm a, MpiComm b)
{
    return !(a == b);
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // comm_Types_hh
