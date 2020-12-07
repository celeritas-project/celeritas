//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file src/comm/ScopedMpiInit.nompi.cc
//---------------------------------------------------------------------------//
#include "ScopedMpiInit.hh"

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Constructor is a null-op
 */
ScopedMpiInit::ScopedMpiInit(int*, char***) {}

//---------------------------------------------------------------------------//
/*!
 * Call MPI finalize on destruction.
 */
ScopedMpiInit::~ScopedMpiInit() = default;

//---------------------------------------------------------------------------//
/*!
 * MPI is disabled for this build.
 */
auto ScopedMpiInit::status() -> Status
{
    return ScopedMpiInit::Status::disabled;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
