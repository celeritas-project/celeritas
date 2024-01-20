//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/MpiType.hh
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
//! Traits class for MPI enumerations
template<class T>
struct MpiType;

#if CELERITAS_USE_MPI
#    define CELER_DEFINE_MPITYPE(T, MPI_ENUM) \
        template<>                            \
        struct MpiType<T>                     \
        {                                     \
            static MPI_Datatype get()         \
            {                                 \
                return MPI_ENUM;              \
            }                                 \
        }
#else
#    define CELER_DEFINE_MPITYPE(T, MPI_ENUM) \
        template<>                            \
        struct MpiType<T>                     \
        {                                     \
            static const char* get()          \
            {                                 \
                return #MPI_ENUM;             \
            }                                 \
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
}  // namespace detail
}  // namespace celeritas
