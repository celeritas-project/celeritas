//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootStepWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <array>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "celeritas/io/RootFileManager.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/user/StepInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write "MC truth" data to ROOT at every step.
 *
 * `TTree::Fill()` is called for each step and thread id, making each ROOT
 * entry a step. Since the ROOT data is stored in branches with primitive types
 * instead of a full struct, no dictionaries are needed for reading the output
 * file.
 *
 * The data can be filtered by defining a `RSWFilter`. This
 * `pair<TStepSelection, TStepData>` informs which data should be filtered
 * (TStepSelection booleans) and by what values (TStepData). If no filtering is
 * needed, simply pass a nullptr to the constructor.
 */
class RootStepWriter final : public StepInterface
{
  public:
    //! Truth step point data; Naming convention must match StepPointStateData
    struct TStepPoint
    {
        size_type volume_id;
        real_type energy;  //!< [MeV]
        real_type time;  //!< [s]
        std::array<real_type, 3> pos;  //!< [cm]
        std::array<real_type, 3> dir;
    };

    //! Truth step data; Naming convention must match StepStateData
    struct TStepData
    {
        size_type event_id;
        size_type track_id;
        size_type parent_id;
        size_type action_id;
        size_type track_step_count;
        int particle;  //!< PDG number
        real_type energy_deposition;  //!< [MeV]
        real_type step_length;  //!< [cm]
        EnumArray<StepPoint, TStepPoint> points;
    };

    //! Filter selection booleans; Naming convention must match TStepData
    struct TStepFilterSelection
    {
        bool event_id{false};
        bool track_id{false};
        bool parent_id{false};
        bool track_step_count{false};
        bool action_id{false};
        bool step_length{false};
        bool particle{false};
        bool energy_deposition{false};
        EnumArray<StepPoint, StepPointSelection> points;

        //! Compare selection with another (use boost pfr?)
        bool operator==(TStepFilterSelection const& other)
        {
            return this->event_id == other.event_id
                   && this->track_id == other.track_id
                   && this->parent_id == other.parent_id
                   && this->track_step_count == other.track_step_count
                   && this->action_id == other.action_id
                   && this->step_length == other.step_length
                   && this->particle == other.particle
                   && this->energy_deposition == other.energy_deposition
                   // Pre-step
                   && this->points[StepPoint::pre].dir
                          == other.points[StepPoint::pre].dir
                   && this->points[StepPoint::pre].pos
                          == other.points[StepPoint::pre].pos
                   && this->points[StepPoint::pre].energy
                          == other.points[StepPoint::pre].energy
                   && this->points[StepPoint::pre].time
                          == other.points[StepPoint::pre].time
                   && this->points[StepPoint::pre].volume_id
                          == other.points[StepPoint::pre].volume_id
                   // Post-step
                   && this->points[StepPoint::post].dir
                          == other.points[StepPoint::post].dir
                   && this->points[StepPoint::post].pos
                          == other.points[StepPoint::post].pos
                   && this->points[StepPoint::post].energy
                          == other.points[StepPoint::post].energy
                   && this->points[StepPoint::post].time
                          == other.points[StepPoint::post].time
                   && this->points[StepPoint::post].volume_id
                          == other.points[StepPoint::post].volume_id;
        }
    };

  public:
    //!@{
    //! \name Type aliases
    using SPRootFileManager = std::shared_ptr<RootFileManager>;
    using SPParticleParams = std::shared_ptr<ParticleParams const>;
    using RSWFilter = std::pair<TStepFilterSelection, TStepData>;
    using UPRSWFilter = std::unique_ptr<RSWFilter>;
    //!@}

    // Construct with RootFileManager, ParticleParams, and data selection
    // UPRSWFilter should be passed as a nullptr if all data should be stored
    RootStepWriter(SPRootFileManager root_manager,
                   SPParticleParams particle_params,
                   StepSelection selection,
                   UPRSWFilter filter_conditions);

    // Set number of entries stored in memory before being flushed to disk
    void set_auto_flush(long num_entries);

    // Process step data on the host and fill step tree
    void execute(StateHostRef const& steps) final;

    // Device execution cannot be implemented
    void execute(StateDeviceRef const&) final
    {
        CELER_NOT_IMPLEMENTED("RootStepWriter is host-only.");
    }

    // Selection of data to be stored
    StepSelection selection() const final { return selection_; }

    // No filter selection is implemented
    Filters filters() const final { return Filters{}; }

  private:
    // Create steps tree based on selection_ booleans
    void make_tree();

    // Verify if current step must be saved
    bool is_selection_valid();

    // Validate correctness of the RSW selection booleans
    void validate_rsw_filter();

  private:
    SPRootFileManager root_manager_;
    SPParticleParams particles_;
    StepSelection selection_;
    detail::RootUniquePtr<TTree> tstep_tree_;
    TStepData tstep_;  // Members are used as refs of the TTree branches
    UPRSWFilter rsw_filter_;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootStepWriter::RootStepWriter(SPRootFileManager,
                                      SPParticleParams,
                                      StepSelection,
                                      UPRSWFilter)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void RootStepWriter::execute(StateHostRef const&)
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
