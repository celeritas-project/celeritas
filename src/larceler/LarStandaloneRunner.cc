//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/
//---------------------------------------------------------------------------//
#include "LarStandaloneRunner.hh"

#include "corecel/Assert.hh"
#include "celeritas/optical/Transporter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with input parameters.
 */
LarStandaloneRunner::LarStandaloneRunner(Input const&)
{
    CELER_NOT_IMPLEMENTED("LarStandaloneRunner");
}

//---------------------------------------------------------------------------//
//! Default destructor
LarStandaloneRunner::~LarStandaloneRunner() = default;

//---------------------------------------------------------------------------//
}  // namespace celeritas
