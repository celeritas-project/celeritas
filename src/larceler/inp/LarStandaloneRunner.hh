//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/inp/LarStandaloneRunner.hh
//---------------------------------------------------------------------------//
#pragma once

#include <limits>
#include <map>
#include <string>

#include "corecel/Types.hh"
#include "celeritas/inp/Control.hh"

namespace celeritas
{
namespace detail
{
struct LarCelerStandaloneConfig;
}
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Input parameters for running LarCelerStandalone optical transport.
 *
 * Variables are copied to a problem input (with classes in the \c
 * celeritas::inp namespace):
 * - \c environment: used in \c system in \c System
 * - \c geometry: saved to \c p.model in \c Model
 * - \c optical_step_iters: saved to \c p.tracking.limits in \c TrackingLimits
 * - \c optical_capacity: saved to \c p.control.optical_capacity in \c Control
 * - \c seed: saved to \c p.control.seed in \c Control
 * - (MAYBE?) \c diagnostics: saved to \c p.diagnostics in \c Problem
 */
struct LarStandaloneRunner
{
    //! Don't limit the number of steps (from TrackingLimits)
    static constexpr size_type unlimited
        = std::numeric_limits<size_type>::max();

    //// DATA ////

    //! Environment variables used for program setup/diagnostic
    std::map<std::string, std::string> environment;

    //! GDML input filename
    std::string geometry;

    //! Step iterations before aborting the optical stepping loop
    size_type optical_step_iters{unlimited};

    //! Optical buffer sizes
    OpticalStateCapacity optical_capacity;

    //! Random number generator seed
    unsigned int seed{};
};

//---------------------------------------------------------------------------//
// Convert from a FHiCL config input
inp::LarStandaloneRunner
from_config(detail::LarCelerStandaloneConfig const& config);

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
