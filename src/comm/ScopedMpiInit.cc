//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file src/comm/ScopedMpiInit.cc
//---------------------------------------------------------------------------//
#include "ScopedMpiInit.hh"

#include <cstdlib>

#include "celeritas_config.h"
#if CELERITAS_USE_MPI
#    include <mpi.h>
#endif

#include "base/Assert.hh"
#include "base/Macros.hh"
#include "base/Stopwatch.hh"
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
            if (CELERITAS_USE_MPI)
            {
                CELER_LOG(info) << "Disabling MPI support since the "
                                   "'CELER_DISABLE_PARALLEL' environment "
                                   "variable is present and non-empty";
            }
            break;
        }
        case Status::uninitialized: {
            Stopwatch get_time;
            CELER_MPI_CALL(MPI_Init(argc, argv));
            status_ = Status::initialized;
            CELER_LOG(debug) << "MPI initialization took " << get_time() << "s";
            break;
        }
        case Status::initialized: {
            CELER_LOG(warning) << "MPI was initialized before calling "
                                  "ScopedMpiInit";
            break;
        }
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
        CELER_MPI_CALL(MPI_Finalize());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Whether MPI has been initialized or disabled.
 *
 * NOTE: This function *cannot* call the CELER_LOG macros because those macros
 * query the status.
 */
auto ScopedMpiInit::status() -> Status
{
    if (!CELERITAS_USE_MPI)
    {
        status_ = Status::disabled;
    }
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
            CELER_MPI_CALL(MPI_Initialized(&result));
            status_ = result ? Status::initialized : Status::uninitialized;
        }
    }
    return status_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
