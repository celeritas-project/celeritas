//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MpiCommunicator.cc
//---------------------------------------------------------------------------//
#include "MpiCommunicator.hh"

#include "corecel/Assert.hh"

#include "ScopedMpiInit.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct a communicator with MPI_COMM_WORLD or null if disabled.
 */
MpiCommunicator MpiCommunicator::comm_default()
{
    if (ScopedMpiInit::status() == ScopedMpiInit::Status::disabled)
        return {};

    return comm_world();
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a native MPI communicator.
 *
 * This will fail with a \c NotConfigured error if MPI is disabled.
 */
MpiCommunicator::MpiCommunicator(MpiComm comm) : comm_(comm)
{
    CELER_EXPECT(comm != detail::mpi_comm_null());
    CELER_VALIDATE(
        ScopedMpiInit::status() == ScopedMpiInit::Status::initialized,
        << "MPI was not initialized (needed to construct a communicator). "
           "Maybe set the environment variable CELER_DISABLE_PARALLEL=1 to "
           "disable externally?");

    // Save rank and size
    CELER_MPI_CALL(MPI_Comm_rank(comm_, &rank_));
    CELER_MPI_CALL(MPI_Comm_size(comm_, &size_));

    CELER_ENSURE(this->rank() >= 0 && this->rank() < this->size());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
