//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Communicator.mpi.cc
//---------------------------------------------------------------------------//
#include "Communicator.hh"

#include "base/Assert.hh"
#include "ScopedMpiInit.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a native MPI communicator.
 */
Communicator::Communicator(MpiComm comm) : comm_(comm)
{
    CELER_EXPECT(comm != detail::MpiCommNull());
    CELER_EXPECT(ScopedMpiInit::status() == ScopedMpiInit::Status::initialized);

    // Save rank and size
    int err;
    err = MPI_Comm_rank(comm_, &rank_);
    CELER_ASSERT(err == MPI_SUCCESS);
    err = MPI_Comm_size(comm_, &size_);
    CELER_ASSERT(err == MPI_SUCCESS);

    CELER_ENSURE(this->rank() >= 0 && this->rank() < this->size());
}

//---------------------------------------------------------------------------//
} // namespace celeritas
