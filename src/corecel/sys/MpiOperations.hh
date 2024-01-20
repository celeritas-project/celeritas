//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MpiOperations.hh
//! \brief MPI parallel functionality
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <type_traits>

#include "celeritas_config.h"

#if CELERITAS_USE_MPI
#    include <mpi.h>
#endif

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"

#include "MpiCommunicator.hh"
#include "detail/MpiType.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPES
//---------------------------------------------------------------------------//
//! MPI reduction operation to perform on the data
enum class Operation
{
    min,
    max,
    sum,
    prod,
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Wait for all processes in this communicator to reach the barrier
inline void barrier(MpiCommunicator const& comm);

//---------------------------------------------------------------------------//
// All-to-all reduction on the data from src to dst
template<class T, std::size_t N>
inline void allreduce(MpiCommunicator const& comm,
                      Operation op,
                      Span<T const, N> src,
                      Span<T, N> dst);

//---------------------------------------------------------------------------//
// All-to-all reduction on the data, in place
template<class T, std::size_t N>
inline void
allreduce(MpiCommunicator const& comm, Operation op, Span<T, N> data);

//---------------------------------------------------------------------------//
// Perform reduction on a fundamental scalar and return the result
template<class T, std::enable_if_t<std::is_fundamental<T>::value, T*> = nullptr>
inline T allreduce(MpiCommunicator const& comm, Operation op, T const src);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
namespace
{
#if CELERITAS_USE_MPI
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
    CELER_ASSERT_UNREACHABLE();
}
#endif
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Wait for all processes in this communicator to reach the barrier.
 */
void barrier(MpiCommunicator const& comm)
{
    if (!comm)
        return;

    CELER_MPI_CALL(MPI_Barrier(comm.mpi_comm()));
}

//---------------------------------------------------------------------------//
/*!
 * All-to-all reduction on the data from src to dst.
 */
template<class T, std::size_t N>
void allreduce(MpiCommunicator const& comm,
               [[maybe_unused]] Operation op,
               Span<T const, N> src,
               Span<T, N> dst)
{
    CELER_EXPECT(src.size() == dst.size());

    if (!comm)
    {
        std::copy(src.begin(), src.end(), dst.begin());
        return;
    }

    CELER_MPI_CALL(MPI_Allreduce(src.data(),
                                 dst.data(),
                                 dst.size(),
                                 detail::MpiType<T>::get(),
                                 to_mpi(op),
                                 comm.mpi_comm()));
}

//---------------------------------------------------------------------------//
/*!
 * All-to-all reduction on the data, in place.
 */
template<class T, std::size_t N>
void allreduce(MpiCommunicator const& comm,
               [[maybe_unused]] Operation op,
               [[maybe_unused]] Span<T, N> data)
{
    if (!comm)
        return;

    CELER_MPI_CALL(MPI_Allreduce(MPI_IN_PLACE,
                                 data.data(),
                                 data.size(),
                                 detail::MpiType<T>::get(),
                                 to_mpi(op),
                                 comm.mpi_comm()));
}

//---------------------------------------------------------------------------//
/*!
 * Perform reduction on a fundamental scalar and return the result.
 */
template<class T, std::enable_if_t<std::is_fundamental<T>::value, T*>>
T allreduce(MpiCommunicator const& comm, Operation op, T const src)
{
    T dst{};
    allreduce(comm, op, Span<T const, 1>{&src, 1}, Span<T, 1>{&dst, 1});
    return dst;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
