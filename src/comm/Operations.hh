//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Operations.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <type_traits>

#include "celeritas_config.h"

#if CELERITAS_USE_MPI
#    include <mpi.h>
#endif

#include "base/Span.hh"
#include "base/Assert.hh"
#include "base/Macros.hh"
#include "detail/MpiTypes.hh"
#include "Communicator.hh"

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
inline void barrier(const Communicator& comm);

//---------------------------------------------------------------------------//
// All-to-all reduction on the data from src to dst
template<class T, std::size_t N>
inline void allreduce(const Communicator& comm,
                      Operation           op,
                      Span<const T, N>    src,
                      Span<T, N>          dst);

//---------------------------------------------------------------------------//
// All-to-all reduction on the data, in place
template<class T, std::size_t N>
inline void allreduce(const Communicator& comm, Operation op, Span<T, N> data);

//---------------------------------------------------------------------------//
// Perform reduction on a fundamental scalar and return the result
template<class T, std::enable_if_t<std::is_fundamental<T>::value, T*> = nullptr>
inline T allreduce(const Communicator& comm, Operation op, const T src);

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
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Wait for all processes in this communicator to reach the barrier.
 */
void barrier(const Communicator& comm)
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
void allreduce(const Communicator&          comm,
               CELER_MAYBE_UNUSED Operation op,
               Span<const T, N>             src,
               Span<T, N>                   dst)
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
void allreduce(const Communicator&          comm,
               CELER_MAYBE_UNUSED Operation op,
               CELER_MAYBE_UNUSED Span<T, N> data)
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
T allreduce(const Communicator& comm, Operation op, const T src)
{
    T dst{};
    allreduce(comm, op, Span<const T, 1>{&src, 1}, Span<T, 1>{&dst, 1});
    return dst;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
