//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file src/comm/ScopedMpiInit.nompi.cc
//---------------------------------------------------------------------------//
#include "ScopedMpiInit.hh"

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
}

//---------------------------------------------------------------------------//
/*!
 * \brief Whether MPI has been initialized
 */
bool ScopedMpiInit::initialized()
{
    return g_initialized;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
