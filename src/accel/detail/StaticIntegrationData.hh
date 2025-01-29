//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/StaticIntegrationData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "accel/LocalTransporter.hh"
#include "accel/SetupOptions.hh"
#include "accel/SharedParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//

// Static global setup options before constructing params
SetupOptions& setup_options();

// Static global Celeritas problem data
SharedParams& shared_params();

// Static THREAD-LOCAL Celeritas state data
LocalTransporter& local_transporter();

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
