//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Communicator.nompi.cc
//---------------------------------------------------------------------------//
#include "Communicator.hh"

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a native MPI communicator
 */
Communicator::Communicator(MpiComm)
{
    throw DebugError("Cannot build a communicator because MPI is disabled");
}

//---------------------------------------------------------------------------//
} // namespace celeritas
