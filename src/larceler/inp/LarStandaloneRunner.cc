//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/inp/LarStandaloneRunner.cc
//---------------------------------------------------------------------------//
#include "LarStandaloneRunner.hh"

#include "larceler/detail/LarCelerConfig.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Convert from a FHiCL config input.
 */
inp::LarStandaloneRunner
from_config(detail::LarCelerStandaloneConfig const& cfg)
{
    LarStandaloneRunner out;

#if 0
    // FIXME: environment config doesn't yet work
    {
        fhicl::ParameterSet const& ps = cfg.environment();
        for (auto const& key : ps.get_names()) {
            out.environment[key] = ps.get<std::string>(key);
        }
    }
#endif

    out.geometry = cfg.geometry();
    if (auto step_iters = cfg.optical_step_iters())
    {
        out.tracking_limits.step_iters = step_iters;
    }

    // Optical capacities
    {
        auto const& ocfg = cfg.optical_capacity();
        out.optical_capacity.primaries = ocfg.primaries();
        out.optical_capacity.tracks = ocfg.tracks();
        out.optical_capacity.generators = ocfg.generators();
    }

    out.seed = cfg.seed();

    return out;
}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
