//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Communicator.nompi.cc
//---------------------------------------------------------------------------//
#include "Communicator.hh"

#include "base/Assert.hh"
#include "ScopedMpiInit.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a "self" MPI communicator: always world size of 1
 */
Communicator Communicator::comm_self()
{
    return Communicator(MpiComm{1});
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a "world" MPI communicator: always global size
 */
Communicator Communicator::comm_world()
{
    return Communicator(MpiComm{2});
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a native MPI communicator
 */
Communicator::Communicator(MpiComm comm) : comm_(comm), rank_(0), size_(1)
{
    REQUIRE(ScopedMpiInit::initialized());

    ENSURE(this->rank() >= 0 && this->rank() < this->size());
}

//---------------------------------------------------------------------------//
/*!
 * \brief Wait for all processes in this communicator to reach the barrier
 */
void Communicator::barrier() const {}

//---------------------------------------------------------------------------//
} // namespace celeritas
