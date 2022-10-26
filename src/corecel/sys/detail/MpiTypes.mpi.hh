//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/MpiTypes.mpi.hh
//---------------------------------------------------------------------------//
#pragma once

#include <mpi.h>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
using MpiComm = MPI_Comm;

inline MpiComm MpiCommNull()
{
    return MPI_COMM_NULL;
}
inline MpiComm MpiCommWorld()
{
    return MPI_COMM_WORLD;
}
inline MpiComm MpiCommSelf()
{
    return MPI_COMM_SELF;
}

template<class T>
struct MpiType;

#define CELER_DEFINE_MPITYPE(T, MPI_ENUM)              \
    template<>                                         \
    struct MpiType<T>                                  \
    {                                                  \
        static MPI_Datatype get() { return MPI_ENUM; } \
    }

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
} // namespace detail
} // namespace celeritas
