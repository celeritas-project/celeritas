//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/System.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <optional>
#include <string>

#include "celeritas/Types.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Set up GPU capabilities and debugging options.
 *
 * Stream sharing and synchronization might be helpful for debugging potential
 * race conditions or improving timing accuracy (at the cost of reducing
 * performance).
 *
 * The CUDA heap and stack sizes may be needed for VecGeom, which has dynamic
 * resource requirements.
 *
 * \todo Move the \c CELER_DEVICE_ASYNC environment variable here
 */
struct Device
{
    //! Per-thread CUDA stack size (ignored if zero) [B]
    size_type stack_size{};
    //! Global dynamic CUDA heap size (ignored if zero) [B]
    size_type heap_size{};

    // TODO: could add preferred device ID, etc.
};

//---------------------------------------------------------------------------//
/*!
 * Set up system parameters defined once at program startup.
 *
 * \todo Add OpenMP options
 * \todo Add MPI options
 * \todo Add Logger verbosity
 */
struct System
{
    //! Environment variables used for program setup/diagnostic
    std::map<std::string, std::string> environment;

    //! Optional: activate GPU
    std::optional<Device> device;
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
