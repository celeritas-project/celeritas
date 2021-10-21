//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleProcessDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Diagnostic.hh"

#include <string>
#include <unordered_map>
#include "physics/base/ParticleParams.hh"
#include "physics/base/PhysicsParams.hh"

using celeritas::Collection;
using celeritas::MemSpace;
using celeritas::Ownership;
using celeritas::size_type;

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Diagnostic class for collecting information on particle and process types.
 *
 * Tallies the particle/process combinations that underwent a discrete
 * interaction at each step.
 */
template<MemSpace M>
class ParticleProcessDiagnostic : public Diagnostic<M>
{
  public:
    //!@{
    //! Type aliases
    using ParamsDataRef = celeritas::ParamsData<Ownership::const_reference, M>;
    using StateDataRef  = celeritas::StateData<Ownership::reference, M>;
    using SPConstParticles = std::shared_ptr<const celeritas::ParticleParams>;
    using SPConstPhysics   = std::shared_ptr<const celeritas::PhysicsParams>;
    //!@}

  public:
    // Construct with shared problem data
    ParticleProcessDiagnostic(const ParamsDataRef& params,
                              SPConstParticles     particles,
                              SPConstPhysics       physics);

    // Particle/model tallied after sampling discrete interaction
    void mid_step(const StateDataRef& states) final;

    // Get particle-process and number of times the interaction occured
    std::unordered_map<std::string, size_type> particle_processes() const;

  private:
    // Shared problem data
    const ParamsDataRef& params_;
    // Shared particle data for getting particle name from particle ID
    SPConstParticles particles_;
    // Shared physics data for getting process name from model ID
    SPConstPhysics physics_;
    // Count of particle/model combinations that underwent discrete interaction
    Collection<size_type, Ownership::value, M> counts_;
};

//---------------------------------------------------------------------------//
// KERNEL LAUNCHER(S)
//---------------------------------------------------------------------------//
/*!
 * Diagnostic kernel launcher
 */
template<MemSpace M>
class ParticleProcessLauncher
{
  public:
    //!@{
    //! Type aliases
    using ParamsDataRef = celeritas::ParamsData<Ownership::const_reference, M>;
    using StateDataRef  = celeritas::StateData<Ownership::reference, M>;
    using ItemsRef      = Collection<size_type, Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION ParticleProcessLauncher(const ParamsDataRef& params,
                                           const StateDataRef&  states,
                                           ItemsRef&            counts);

    //! Create track views and tally particle/processes
    inline CELER_FUNCTION void operator()(ThreadId tid) const;

  private:
    const ParamsDataRef& params_;
    const StateDataRef&  states_;
    ItemsRef&            counts_;
};

void count_particle_process(
    const celeritas::ParamsHostRef&                             params,
    const celeritas::StateHostRef&                              states,
    Collection<size_type, Ownership::reference, MemSpace::host> counts);

void count_particle_process(
    const celeritas::ParamsDeviceRef&                             params,
    const celeritas::StateDeviceRef&                              states,
    Collection<size_type, Ownership::reference, MemSpace::device> counts);

#if !CELERITAS_USE_CUDA
inline void count_particle_process(
    const celeritas::ParamsDeviceRef&                             params,
    const celeritas::StateDeviceRef&                              states,
    Collection<size_type, Ownership::reference, MemSpace::device> counts)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

} // namespace demo_loop

#include "ParticleProcessDiagnostic.i.hh"
