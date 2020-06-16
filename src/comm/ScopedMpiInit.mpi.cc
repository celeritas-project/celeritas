//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file src/comm/ScopedMpiInit.mpi.cc
//---------------------------------------------------------------------------//
#include "ScopedMpiInit.hh"

#include <mpi.h>
#include "base/Assert.hh"

namespace
{
bool g_initialized = false;
}

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * \brief Construct with argc/argv references
 */
ScopedMpiInit::ScopedMpiInit(int* argc, char*** argv)
{
    REQUIRE((argc == nullptr) == (argv == nullptr));
    REQUIRE(!ScopedMpiInit::initialized());

    int err = MPI_Init(argc, argv);
    CHECK(err == MPI_SUCCESS);

    g_initialized = true;
    ENSURE(ScopedMpiInit::initialized());
}

//---------------------------------------------------------------------------//
/*!
 * \brief Call MPI finalize on destruction
 */
ScopedMpiInit::~ScopedMpiInit()
{
    g_initialized = false;
    int err       = MPI_Finalize();
    ENSURE(err == MPI_SUCCESS);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Whether MPI has been initialized
 */
bool ScopedMpiInit::initialized()
{
    if (!g_initialized)
    {
        // Allow for the case where another application has already initialized
        // MPI.
        int result = -1;
        int err    = MPI_Initialized(&result);
        CHECK(err == MPI_SUCCESS);
        g_initialized = static_cast<bool>(result);
    }
    return g_initialized;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
