//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MpiCommunicator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "MpiCommunicator.hh"
#include "detail/MpiCommunicatorImpl.hh"
#include "detail/MpiCommunicatorImpl.hh"  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Abstraction of an MPI communicator.
 *
 * A "null" communicator (the default) does not use MPI calls and can be
 * constructed without calling \c MPI_Init or having MPI compiled. It will act
 * like \c MPI_Comm_Self but will not actually use MPI calls.
 *
 * TODO: drop \c comm_ prefix from static helpers
 */
class MpiCommunicator
{
  public:
    //!@{
    //! \name Type aliases
    using MpiComm = detail::MpiComm;
    //!@}

  public:
    // Construct a communicator with MPI_COMM_SELF
    inline static MpiCommunicator comm_self();

    // Construct a communicator with MPI_COMM_WORLD
    inline static MpiCommunicator comm_world();

    // Construct a communicator with MPI_COMM_WORLD or null if disabled
    static MpiCommunicator comm_default();

    //// CONSTRUCTORS ////

    // Construct with a null communicator (MPI is disabled)
    MpiCommunicator() = default;

    // Construct with a native MPI communicator
    explicit MpiCommunicator(MpiComm comm);

    //// ACCESSORS ////

    //! Get the MPI communicator for low-level MPI calls
    MpiComm mpi_comm() const { return comm_; }

    //! Get the local process ID
    int rank() const { return rank_; }

    //! Get the number of total processors
    int size() const { return size_; }

    //! True if non-null communicator
    explicit operator bool() const { return comm_ != detail::mpi_comm_null(); }

  private:
    MpiComm comm_ = detail::mpi_comm_null();
    int rank_ = 0;
    int size_ = 1;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
//! Construct a communicator with MPI_COMM_SELF
MpiCommunicator MpiCommunicator::comm_self()
{
    return MpiCommunicator{detail::mpi_comm_self()};
}

//---------------------------------------------------------------------------//
//! Construct a communicator with MPI_COMM_WORLD
MpiCommunicator MpiCommunicator::comm_world()
{
    return MpiCommunicator{detail::mpi_comm_world()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
