//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/StaticIntegrationData.cc
//---------------------------------------------------------------------------//
#include "StaticIntegrationData.hh"

#include <G4RunManager.hh>
#include <G4Threading.hh>

#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Static global setup options before constructing params.
 */
SetupOptions& setup_options()
{
    static SetupOptions so;
    return so;
}

//---------------------------------------------------------------------------//
/*!
 * Static global Celeritas problem data.
 */
SharedParams& shared_params()
{
    static SharedParams sp;
    return sp;
}

//---------------------------------------------------------------------------//
/*!
 * Static THREAD-LOCAL Celeritas state data.
 */
LocalTransporter& local_transporter()
{
    static G4ThreadLocal LocalTransporter lt;
    return lt;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
