//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/MpiCommunicatorImpl.hh
//! \brief Type definitions for MPI parallel operations (MPI optional)
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#if CELERITAS_USE_MPI
#    include <mpi.h>
#    define CELER_IF_MPI_ELSE(TRUE, FALSE) TRUE
#else
#    define CELER_IF_MPI_ELSE(TRUE, FALSE) FALSE
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Traits class for MPI enumerations
template<class T>
struct MpiType;

#if CELERITAS_USE_MPI
#    define CELER_DEFINE_MPITYPE(T, MPI_ENUM) \
        template<> \
        struct MpiType<T> \
        { \
            static inline MPI_Datatype value = MPI_ENUM; \
        }
#else
#    define CELER_DEFINE_MPITYPE(T, MPI_ENUM) \
        template<> \
        struct MpiType<T> \
        { \
            static inline const char* value = #MPI_ENUM; \
        }
#endif

CELER_DEFINE_MPITYPE(bool, MPI_CXX_BOOL);
CELER_DEFINE_MPITYPE(char, MPI_CHAR);
CELER_DEFINE_MPITYPE(unsigned char, MPI_UNSIGNED_CHAR);
CELER_DEFINE_MPITYPE(short, MPI_SHORT);
CELER_DEFINE_MPITYPE(unsigned short, MPI_UNSIGNED_SHORT);
CELER_DEFINE_MPITYPE(int, MPI_INT);
CELER_DEFINE_MPITYPE(unsigned int, MPI_UNSIGNED);
CELER_DEFINE_MPITYPE(long, MPI_LONG);
CELER_DEFINE_MPITYPE(unsigned long, MPI_UNSIGNED_LONG);
CELER_DEFINE_MPITYPE(long long, MPI_LONG_LONG);
CELER_DEFINE_MPITYPE(unsigned long long, MPI_UNSIGNED_LONG_LONG);
CELER_DEFINE_MPITYPE(float, MPI_FLOAT);
CELER_DEFINE_MPITYPE(double, MPI_DOUBLE);
CELER_DEFINE_MPITYPE(long double, MPI_LONG_DOUBLE);

#undef CELER_DEFINE_MPITYPE

//---------------------------------------------------------------------------//
// MPI COMM WRAPPER
//---------------------------------------------------------------------------//
//!@{
//! Mock MPI communicator
struct MockMpiComm
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

//! Opaque MPI communicator
using MpiComm = CELER_IF_MPI_ELSE(MPI_Comm, MockMpiComm);

//! Used for invalid comm handles
inline MpiComm mpi_comm_null()
{
    return CELER_IF_MPI_ELSE(MPI_COMM_NULL, {0});
}

//! Communicator for all processes
inline MpiComm mpi_comm_world()
{
    return CELER_IF_MPI_ELSE(MPI_COMM_WORLD, {1});
}

//! Communicator with only the local process
inline MpiComm mpi_comm_self()
{
    return CELER_IF_MPI_ELSE(MPI_COMM_SELF, {-1});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
