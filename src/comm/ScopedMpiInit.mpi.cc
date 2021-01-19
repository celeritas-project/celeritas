//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file src/comm/ScopedMpiInit.mpi.cc
//---------------------------------------------------------------------------//
#include "ScopedMpiInit.hh"

#include <cstdlib>
#include <mpi.h>
#include "base/Assert.hh"
#include "base/Macros.hh"
#include "base/Stopwatch.hh"
#include "Logger.hh"
#include "Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
ScopedMpiInit::Status ScopedMpiInit::status_
    = ScopedMpiInit::Status::uninitialized;

//---------------------------------------------------------------------------//
/*!
 * Construct with argc/argv references.
 */
ScopedMpiInit::ScopedMpiInit(int* argc, char*** argv)
{
    CELER_EXPECT((argc == nullptr) == (argv == nullptr));

    switch (ScopedMpiInit::status())
    {
        case Status::disabled: {
            CELER_LOG(info)
                << "Disabling MPI support since the 'CELER_DISABLE_PARALLEL' "
                   "environment variable is present and non-empty";
            break;
        }
        case Status::uninitialized: {
            Stopwatch get_time;
            int       err = MPI_Init(argc, argv);
            CELER_ASSERT(err == MPI_SUCCESS);
            status_ = Status::initialized;
            CELER_LOG(debug) << "MPI initialization took " << get_time() << "s";
            break;
        }
        case Status::initialized: {
            CELER_LOG(warning) << "MPI was initialized before calling "
                                  "ScopedMpiInit";
            break;
        }
        default:
            CELER_ASSERT_UNREACHABLE();
    }
    CELER_ENSURE(status_ != Status::uninitialized);
}

//---------------------------------------------------------------------------//
/*!
 * Call MPI finalize on destruction.
 */
ScopedMpiInit::~ScopedMpiInit()
{
    if (status_ == Status::initialized)
    {
        status_ = Status::uninitialized;
        int err = MPI_Finalize();
        CELER_ENSURE(err == MPI_SUCCESS);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Whether MPI has been initialized or disabled.
 */
auto ScopedMpiInit::status() -> Status
{
    if (CELER_UNLIKELY(status_ == Status::uninitialized))
    {
        const char* disable = std::getenv("CELER_DISABLE_PARALLEL");
        if (disable && disable[0] != '\0')
        {
            // MPI is disabled via an environment variable.
            status_ = Status::disabled;
        }
        else
        {
            // Allow for the case where another application has already
            // initialized MPI.
            int result = -1;
            int err    = MPI_Initialized(&result);
            CELER_ASSERT(err == MPI_SUCCESS);
            status_ = result ? Status::initialized : Status::uninitialized;
        }
    }
    return status_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
