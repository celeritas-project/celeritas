//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MpiCommunicator.hh
//! \note Including this file requires linking against the corecel_mpi target
//---------------------------------------------------------------------------//
#pragma once

#include "detail/MpiCommunicatorImpl.hh"  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Wrap an MPI communicator.
 *
 * This class uses \c ScopedMpiInit to determine whether MPI is available
 * and enabled. As many instances as desired can be created, but Celeritas by
 * default will share the instance returned by \c comm_world , which defaults
 * to \c MPI_COMM_WORLD if MPI has been initialized, or a "self" comm if it has
 * not.
 *
 * A "null" communicator (the default) does not use MPI calls and can be
 * constructed without calling \c MPI_Init or having MPI compiled. It will act
 * like \c MPI_Comm_Self but will not actually use MPI calls.
 *
 * \note This does not perform any copying or freeing of MPI communiators.
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
    inline static MpiCommunicator self();

    // Construct a communicator with MPI_COMM_WORLD
    inline static MpiCommunicator world();

    // Construct a communicator with MPI_COMM_WORLD or null if disabled
    static MpiCommunicator world_if_enabled();

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
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Shared "world" Celeritas communicator
MpiCommunicator const& comm_world();

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct a communicator with MPI_COMM_SELF.
 *
 * Each process sees itself as rank zero in size zero, thus operating
 * independently.
 */
MpiCommunicator MpiCommunicator::self()
{
    return MpiCommunicator{detail::mpi_comm_self()};
}

//---------------------------------------------------------------------------//
/*!
 * Construct a communicator with MPI_COMM_WORLD.
 *
 * Each process sees every other process in MPI's domain, operating at the
 * maximum level of interprocess cooperation.
 */
MpiCommunicator MpiCommunicator::world()
{
    return MpiCommunicator{detail::mpi_comm_world()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
