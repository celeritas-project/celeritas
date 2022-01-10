//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StepDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Diagnostic.hh"

#include <string>
#include <unordered_map>
#include <vector>
#include "physics/base/ParticleParams.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Storage for collecting information on steps per track.
 */
template<Ownership W, MemSpace M>
struct StepDiagnosticData
{
    //// TYPES ////

    using size_type = celeritas::size_type;
    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;
    template<class T>
    using Items = celeritas::Collection<T, W, M>;

    //// DATA ////

    StateItems<size_type> steps;  //!< Current step count for each track
    Items<size_type>      counts; //!< Bin tracks by particle and step count
    size_type             num_bins;
    size_type             num_particles;

    //// METHODS ////

    //! Whether the data is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return !steps.empty() && num_bins > 0 && num_particles > 0
               && counts.size() == num_bins * num_particles;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    StepDiagnosticData& operator=(StepDiagnosticData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        steps         = other.steps;
        counts        = other.counts;
        num_bins      = other.num_bins;
        num_particles = other.num_particles;
        return *this;
    }
};

template<MemSpace M>
using StepDiagnosticDataRef = StepDiagnosticData<Ownership::reference, M>;

//---------------------------------------------------------------------------//
/*!
 * Diagnostic class for getting the distribution of the number of steps per
 * track for each particle type.
 */
template<MemSpace M>
class StepDiagnostic : public Diagnostic<M>
{
  public:
    //!@{
    //! Type aliases
    using size_type        = celeritas::size_type;
    using SPConstParticles = std::shared_ptr<const celeritas::ParticleParams>;
    using ParamsDataRef = celeritas::ParamsData<Ownership::const_reference, M>;
    using StateDataRef  = celeritas::StateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared problem data and upper bound on steps per track
    StepDiagnostic(const ParamsDataRef& params,
                   SPConstParticles     particles,
                   size_type            num_tracks,
                   size_type            max_steps);

    // Number of steps per track, tallied before post-processing
    void mid_step(const StateDataRef& states) final;

    // Get distribution of steps per track for each particle type
    std::unordered_map<std::string, std::vector<size_type>> steps();

  private:
    // Shared problem data
    const ParamsDataRef& params_;
    // Shared particle data for getting particle name from particle ID
    SPConstParticles particles_;
    // Data for finding distribution of steps per track
    StepDiagnosticData<Ownership::value, M> data_;
};

//---------------------------------------------------------------------------//
// KERNEL LAUNCHER(S)
//---------------------------------------------------------------------------//
/*!
 * Step diagnostic kernel launcher.
 */
template<MemSpace M>
class StepLauncher
{
  public:
    //!@{
    //! Type aliases
    using size_type     = celeritas::size_type;
    using ThreadId      = celeritas::ThreadId;
    using ParamsDataRef = celeritas::ParamsData<Ownership::const_reference, M>;
    using StateDataRef  = celeritas::StateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION StepLauncher(const ParamsDataRef&     params,
                                const StateDataRef&      states,
                                StepDiagnosticDataRef<M> data);

    // Create track views and tally steps per track
    inline CELER_FUNCTION void operator()(ThreadId tid) const;

  private:
    const ParamsDataRef&     params_;
    const StateDataRef&      states_;
    StepDiagnosticDataRef<M> data_;
};

void count_steps(const celeritas::ParamsHostRef&       params,
                 const celeritas::StateHostRef&        states,
                 StepDiagnosticDataRef<MemSpace::host> data);

void count_steps(const celeritas::ParamsDeviceRef&       params,
                 const celeritas::StateDeviceRef&        states,
                 StepDiagnosticDataRef<MemSpace::device> data);

#if !CELERITAS_USE_CUDA
inline void count_steps(const celeritas::ParamsDeviceRef&,
                        const celeritas::StateDeviceRef&,
                        StepDiagnosticDataRef<MemSpace::device>)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

} // namespace demo_loop

#include "StepDiagnostic.i.hh"
