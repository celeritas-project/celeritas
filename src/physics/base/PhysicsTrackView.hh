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
#include "physics/base/Units.hh"
#include "physics/grid/GridIdFinder.hh"
#include "physics/grid/PhysicsGridCalculator.hh"
#include "physics/material/MaterialView.hh"
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
    using Initializer_t = PhysicsTrackInitializer;
    using PhysicsParamsPointers
        = PhysicsParamsData<Ownership::const_reference, MemSpace::native>;
    using PhysicsStatePointers
        = PhysicsStateData<Ownership::reference, MemSpace::native>;
    using MevEnergy   = units::MevEnergy;
    using ModelFinder = GridIdFinder<MevEnergy, ModelId>;
    //!@}

  public:
    // Construct from "dynamic" state and "static" particle definitions
    inline CELER_FUNCTION PhysicsTrackView(const PhysicsParamsPointers& params,
                                           const PhysicsStatePointers&  states,
                                           ParticleId particle,
                                           MaterialId material,
                                           ThreadId   id);

    // Initialize the track view
    inline CELER_FUNCTION PhysicsTrackView& operator=(const Initializer_t&);

    // Set the remaining MFP to interaction
    inline CELER_FUNCTION void interaction_mfp(real_type);

    // Set the physics step length
    inline CELER_FUNCTION void step_length(real_type);

    // Set the total (process-integrated) macroscopic xs
    inline CELER_FUNCTION void macro_xs(real_type);

    // Select a model for the current interaction (or {} for no interaction)
    inline CELER_FUNCTION void model_id(ModelId);

    //// DYNAMIC PROPERTIES (pure accessors, free) ////

    // Whether the remaining MFP has been calculated
    CELER_FORCEINLINE_FUNCTION bool has_interaction_mfp() const;

    // Remaining MFP to interaction [1]
    CELER_FORCEINLINE_FUNCTION real_type interaction_mfp() const;

    // Maximum step length [cm]
    CELER_FORCEINLINE_FUNCTION real_type step_length() const;

    // Total (process-integrated) macroscopic xs [cm^-1]
    CELER_FORCEINLINE_FUNCTION real_type macro_xs() const;

    // Selected model if interacting
    CELER_FORCEINLINE_FUNCTION ModelId model_id() const;

    //// PROCESSES (depend on particle type and possibly material) ////

    // Number of processes that apply to this track
    inline CELER_FUNCTION ParticleProcessId::value_type
                          num_particle_processes() const;

    // Process ID for the given within-particle process index
    inline CELER_FUNCTION ProcessId process(ParticleProcessId) const;

    // Get table, null if not present for this particle/material/type
    inline CELER_FUNCTION ValueGridId value_grid(ValueGridType table,
                                                 ParticleProcessId) const;

    // Get hardwired model, null if not present
    inline CELER_FUNCTION ModelId hardwired_model(MevEnergy energy,
                                                  ParticleProcessId) const;

    // Models that apply to the given process ID
    inline CELER_FUNCTION
        ModelFinder make_model_finder(ParticleProcessId) const;

    //// STATIC FUNCTIONS (depend only on params data) ////

    // Calculate scaled step range
    inline CELER_FUNCTION real_type range_to_step(real_type range) const;

    // Calculate macroscopic cross section on the fly for the given model
    inline CELER_FUNCTION real_type calc_xs_otf(ModelId       model,
                                                MaterialView& material,
                                                MevEnergy     energy) const;

    // Construct a grid calculator from a physics table
    inline CELER_FUNCTION
        PhysicsGridCalculator make_calculator(ValueGridId) const;

    //// SCRATCH SPACE ////

    // Access scratch space for particle-process cross section calculations
    inline CELER_FUNCTION real_type& per_process_xs(ParticleProcessId);
    inline CELER_FUNCTION real_type  per_process_xs(ParticleProcessId) const;

    //// HACKS ////

    // Process ID for photoelectric effect
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
