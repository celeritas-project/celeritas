//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Operations.mpi.hh
//---------------------------------------------------------------------------//
#pragma once

#include <mpi.h>
#include "base/Assert.hh"
#include "MpiTypes.hh"

namespace celeritas
{
namespace detail
{
namespace
{
inline MPI_Op to_mpi(Operation op)
{
    switch (op)
    {
        // clang-format off
        case Operation::min:  return MPI_MIN;
        case Operation::max:  return MPI_MAX;
        case Operation::sum:  return MPI_SUM;
        case Operation::prod: return MPI_PROD;
            // clang-format on
    }
    CHECK_UNREACHABLE;
}
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Wait for all processes in this communicator to reach the barrier.
 */
inline void barrier(const Communicator& comm)
{
    REQUIRE(comm);
    int err = MPI_Barrier(comm.mpi_comm());
    ENSURE(err == MPI_SUCCESS);
}

//---------------------------------------------------------------------------//
/*!
 * All-to-all reduction on the data from src to dst.
 */
template<class T>
inline void
allreduce(const Communicator& comm, Operation op, Span<const T> src, Span<T> dst)
{
    REQUIRE(comm);
    REQUIRE(src.size() == dst.size());

    int err = MPI_Allreduce(src.data(),
                            dst.data(),
                            dst.size(),
                            detail::MpiType<T>::get(),
                            to_mpi(op),
                            comm.mpi_comm());
    ENSURE(err == MPI_SUCCESS);
}

//---------------------------------------------------------------------------//
/*!
 * All-to-all reduction on the data, in place.
 */
template<class T>
inline void allreduce(const Communicator& comm, Operation op, Span<T> data)
{
    REQUIRE(comm);

    int err = MPI_Allreduce(MPI_IN_PLACE,
                            data.data(),
                            data.size(),
                            detail::MpiType<T>::get(),
                            to_mpi(op),
                            comm.mpi_comm());
    ENSURE(err == MPI_SUCCESS);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
