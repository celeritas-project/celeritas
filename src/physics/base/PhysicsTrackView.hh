//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "PhysicsInterface.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/material/Types.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Physics data for a track.
 *
 * The physics track view provides an interface for data and operations
 * common to most processes and models.
 */
class PhysicsTrackView
{
  public:
    //!@{
    //! Type aliases
    using SpanConstProcessId  = Span<const ProcessId>;
    using PhysicsGridPointers = XsGridPointers;
    using Initializer_t       = PhysicsTrackInitializer;
    //!@}

  public:
    // Construct from "dynamic" state and "static" particle definitions
    inline CELER_FUNCTION PhysicsTrackView(const PhysicsParamsPointers& params,
                                           const PhysicsStatePointers&  states,
                                           ParticleId particle,
                                           MaterialId material,
                                           ThreadId   id);

    // Initialize the track view
    PhysicsTrackView& operator=(const Initializer_t&);

    // Set the remaining MFP to interaction
    inline CELER_FUNCTION void interaction_mfp(real_type);

    // Set the physics step length
    inline CELER_FUNCTION void step_length(real_type);

    // Select a model for the current interaction (or {} for no interaction)
    inline CELER_FUNCTION void model_id(ModelId);

    //// DYNAMIC PROPERTIES (pure accessors, free) ////

    // Whether the remaining MFP has been calculated
    CELER_FORCEINLINE_FUNCTION bool has_interaction_mfp() const;

    // Remaining MFP to interaction
    CELER_FORCEINLINE_FUNCTION real_type interaction_mfp() const;

    // Maximum step length
    CELER_FORCEINLINE_FUNCTION real_type step_length() const;

    // Selected model if interacting
    CELER_FORCEINLINE_FUNCTION ModelId model_id() const;

    //// PROCESSES (depend on particle type and possibly material) ////

    // Number of processes that apply to this track
    inline CELER_FUNCTION ParticleProcessId::value_type
                          num_particle_processes() const;

    // Process ID for the given within-particle process index
    inline CELER_FUNCTION ProcessId process(ParticleProcessId) const;

    // Get table, null if not present for this particle/material/type
    inline CELER_FUNCTION const PhysicsGridPointers*
    table(PhysicsTableType table, ParticleProcessId) const;

    // Models that apply to the given process ID
    inline CELER_FUNCTION const ModelGroup& models(ParticleProcessId) const;

    //// STATIC FUNCTIONS (depend only on params data) ////

    // Calculate scaled step range
    inline CELER_FUNCTION real_type range_to_step(real_type range) const;

    //// SCRATCH SPACE ////

    // Access scratch space for particle-process cross section calculations
    inline CELER_FUNCTION real_type& per_process_xs(ParticleProcessId);
    inline CELER_FUNCTION real_type  per_process_xs(ParticleProcessId) const;

    //// HACKS ////

    // Process ID for photoelectric effect if Livermore model is in use
    inline CELER_FUNCTION ProcessId photoelectric_process_id() const;

    // Process ID for positron annihilation
    inline CELER_FUNCTION ProcessId eplusgg_process_id() const;

  private:
    const PhysicsParamsPointers& params_;
    const PhysicsStatePointers&  states_;
    const ParticleId             particle_;
    const MaterialId             material_;
    const ThreadId               thread_;

    //// IMPLEMENTATION HELPER FUNCTIONS ////

    CELER_FORCEINLINE_FUNCTION PhysicsTrackState& state();
    CELER_FORCEINLINE_FUNCTION const PhysicsTrackState& state() const;
    CELER_FORCEINLINE_FUNCTION const ProcessGroup& process_group() const;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "PhysicsTrackView.i.hh"
