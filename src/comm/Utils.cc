//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.cuda.cc
//---------------------------------------------------------------------------//
#include "Utils.hh"

#include "celeritas_config.h"
#if CELERITAS_USE_CUDA
#    include <cuda_runtime_api.h>
#endif
#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Initialize device in a round-robin fashion from a communicator
void initialize_device(const Communicator& comm)
{
#if CELERITAS_USE_CUDA
    // Get number of devices
    int num_devices = -1;
    CELER_CUDA_CALL(cudaGetDeviceCount(&num_devices));
    CHECK(num_devices > 0);

    // Set device based on communicator
    int device_id = comm.rank() % num_devices;
    CELER_CUDA_CALL(cudaSetDevice(device_id));
#else
    (void)sizeof(comm);
#endif
}

//---------------------------------------------------------------------------//
} // namespace celeritas
