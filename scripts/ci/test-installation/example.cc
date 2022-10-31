//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file scripts/ci/test-installation/example.cc
//---------------------------------------------------------------------------//
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedMpiInit.hh"

int main(int argc, char* argv[])
{
    using celeritas::MpiCommunicator;
    using celeritas::ScopedMpiInit;

    // Initialize MPI
    ScopedMpiInit   scoped_mpi(&argc, &argv);
    MpiCommunicator comm
        = (ScopedMpiInit::status() == ScopedMpiInit::Status::disabled
               ? MpiCommunicator{}
               : MpiCommunicator::comm_world());

    // Initialize GPU
    celeritas::activate_device(celeritas::make_device(comm));

    CELER_LOG(info) << "This example doesn't do anything useful! Sorry!";
    return 0;
}
