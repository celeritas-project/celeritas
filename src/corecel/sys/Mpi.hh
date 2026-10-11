//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Mpi.hh
//! \brief MPI parallel functionality
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <type_traits>

#include "corecel/Config.hh"

#if CELERITAS_USE_MPI
#    include <mpi.h>
#endif

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"

#include "MpiCommunicator.hh"

#include "detail/MpiImpl.hh"

//---------------------------------------------------------------------------//
// MACROS
//---------------------------------------------------------------------------//
/*!
 * \def CELER_MPI_CALL
 *
 * When MPI support is enabled, execute the wrapped statement and throw a
 * RuntimeError if it fails. If MPI is disabled, throw an unconfigured
 * assertion.
 *
 * \note A file that uses this macro must include \c mpi.h.
 */
#if CELERITAS_USE_MPI
#    define CELER_MPI_CALL(STATEMENT) \
        do \
        { \
            int mpi_result_ = (STATEMENT); \
            if (CELER_UNLIKELY(mpi_result_ != MPI_SUCCESS)) \
            { \
                CELER_RUNTIME_FAIL_( \
                    "MPI", \
                    ::celeritas::detail::mpi_error_to_string(mpi_result_), \
                    #STATEMENT); \
            } \
        } while (0)
#else
#    define CELER_MPI_CALL(STATEMENT) \
        do \
        { \
            CELER_NOT_CONFIGURED("MPI"); \
        } while (0)
#endif

namespace celeritas
{
namespace detail
{

//---------------------------------------------------------------------------//
/*!
 * Get an MPI error string.
 */
std::string mpi_error_to_string(int errorcode)
{
#if CELERITAS_USE_MPI
    std::string error_string;
    error_string.resize(MPI_MAX_ERROR_STRING);
    int length = 0;
    int new_errorcode = 0;
    new_errorcode = MPI_Error_string(errorcode, &error_string.front(), &length);
    if (new_errorcode != 0)
    {
        // Even the string call failed... just return the code
        return std::tostring(errorcode);
    }
    error_string.resize(length);
    return error_string;
#else
    CELER_DISCARD(errorcode);
    CELER_NOT_CONFIGURED("MPI");
#endif
}
}  // namespace detail

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
    size_
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
inline void allreduce(
    MpiCommunicator const& comm, Operation op, Span<T, N> data);

//---------------------------------------------------------------------------//
// Perform reduction on a fundamental scalar and return the result
template<class T, std::enable_if_t<std::is_fundamental<T>::value, bool> = true>
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
        default: CELER_ASSERT_UNREACHABLE();
            // clang-format on
    }
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
                                 detail::MpiType<T>::value,
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
                                 detail::MpiType<T>::value,
                                 to_mpi(op),
                                 comm.mpi_comm()));
}

//---------------------------------------------------------------------------//
/*!
 * Perform reduction on a fundamental scalar and return the result.
 */
template<class T, std::enable_if_t<std::is_fundamental<T>::value, bool>>
T allreduce(MpiCommunicator const& comm, Operation op, T const src)
{
    T dst{};
    allreduce(comm, op, Span<T const, 1>{&src, 1}, Span<T, 1>{&dst, 1});
    return dst;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
