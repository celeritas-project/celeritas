//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/MpiCommunicatorImpl.hh
//! \brief Type definitions for MPI parallel operations (MPI optional)
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"

#if CELERITAS_USE_MPI
#    include <mpi.h>
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
#if CELERITAS_USE_MPI

//! Opaque MPI communicator
using MpiComm = MPI_Comm;

#else

//!@{
//! Mock MPI communicator
struct MpiComm
{
    int value_;
};

constexpr inline bool operator==(MpiComm a, MpiComm b)
{
    return a.value_ == b.value_;
}

constexpr inline bool operator!=(MpiComm a, MpiComm b)
{
    return !(a == b);
}
//!@}

#endif

//---------------------------------------------------------------------------//
//! Used for invalid comm handles
inline MpiComm mpi_comm_null()
{
#if CELERITAS_USE_MPI
    return MPI_COMM_NULL;
#else
    return {0};
#endif
}

//! Communicator for all processes
inline MpiComm mpi_comm_world()
{
#if CELERITAS_USE_MPI
    return MPI_COMM_WORLD;
#else
    return {1};
#endif
}

//! Communicator with only the local process
inline MpiComm mpi_comm_self()
{
#if CELERITAS_USE_MPI
    return MPI_COMM_SELF;
#else
    return {-1};
#endif
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
