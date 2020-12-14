//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Communicator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "detail/MpiTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Abstraction of an MPI communicator.
 *
 * A "null" communicator (the default) does not use MPI calls and can be
 * constructed without calling \c MPI_Init or having MPI compiled. It will act
 * like \c MPI_Comm_Self but will not actually use MPI calls.
 */
class Communicator
{
  public:
    //!@{
    //! Type aliases
    using MpiComm = detail::MpiComm;
    //!@}

  public:
    // Construct a communicator with MPI_COMM_SELF
    inline static Communicator comm_self();

    // Construct a communicator with MPI_COMM_WORLD
    inline static Communicator comm_world();

    //// CONSTRUCTORS ////

    // Construct with a null communicator (MPI is disabled)
    Communicator() = default;

    // Construct with a native MPI communicator
    explicit Communicator(MpiComm comm);

    //// ACCESSORS ////

    //! Get the MPI communicator for low-level MPI calls
    MpiComm mpi_comm() const { return comm_; }

    //! Get the local process ID
    int rank() const { return rank_; }

    //! Get the number of total processors
    int size() const { return size_; }

    //! True if non-null communicator
    explicit operator bool() const { return comm_ != detail::MpiCommNull(); }

  private:
    MpiComm comm_ = detail::MpiCommNull();
    int     rank_ = 0;
    int     size_ = 1;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//
//! Construct a communicator with MPI_COMM_SELF
Communicator Communicator::comm_self()
{
    return Communicator{detail::MpiCommSelf()};
}

//---------------------------------------------------------------------------//
//! Construct a communicator with MPI_COMM_WORLD
Communicator Communicator::comm_world()
{
    return Communicator{detail::MpiCommWorld()};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
