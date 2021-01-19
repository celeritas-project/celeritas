//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Operations.i.hh
//---------------------------------------------------------------------------//
#include "Operations.hh"

#include <algorithm>
#include "celeritas_config.h"
#include "base/Assert.hh"
#if CELERITAS_USE_MPI
#    include "detail/Operations.mpi.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Wait for all processes in this communicator to reach the barrier.
 */
void barrier(const Communicator& comm)
{
    if (!comm)
        return;

#if CELERITAS_USE_MPI
    return detail::barrier(comm);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * All-to-all reduction on the data from src to dst.
 */
template<class T, std::size_t N>
void allreduce(const Communicator& comm,
               Operation           op,
               Span<const T, N>    src,
               Span<T, N>          dst)
{
    CELER_EXPECT(src.size() == dst.size());

    if (!comm)
    {
        std::copy(src.begin(), src.end(), dst.begin());
        return;
    }

#if CELERITAS_USE_MPI
    return detail::allreduce(comm, op, Span<const T>(src), Span<T>(dst));
#else
    (void)sizeof(op);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * All-to-all reduction on the data, in place.
 */
template<class T, std::size_t N>
void allreduce(const Communicator& comm, Operation op, Span<T, N> data)
{
    if (!comm)
        return;

#if CELERITAS_USE_MPI
    return detail::allreduce(comm, op, Span<T>(data));
#else
    (void)sizeof(data);
    (void)sizeof(op);
#endif
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
