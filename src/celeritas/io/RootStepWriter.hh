//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
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
 */
class RootStepWriter final : public StepInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPRootFileManager = std::shared_ptr<RootFileManager>;
    using SPParticleParams  = std::shared_ptr<const ParticleParams>;
    //!@}

    // Construct with RootFileManager and ParticleParams
    explicit RootStepWriter(SPRootFileManager root_manager,
                            SPParticleParams  particle_params,
                            StepSelection     selection,
                            Filters           filters);

    // Set number of entries stored in memory before being flushed to disk
    void set_auto_flush(long num_entries);

    // Process step data on the host and fill step tree
    void execute(StateHostRef const& steps) final;

    // Not implemented
    void execute(StateDeviceRef const&) final {}

    // Selection of data to be stored
    StepSelection selection() const final { return selection_; }

    // Selection of filters for the stored data
    Filters filters() const final { return filters_; }

  private:
    // Create steps tree based on selection_ booleans
    void make_tree();

  private:
    template<class T>
    using TRootUP = detail::TRootUniquePtr<T>;

    //// INPUT ////
    SPRootFileManager root_manager_;
    SPParticleParams  particles_;
    StepSelection     selection_;
    Filters           filters_; // TODO: add

    //// DATA ////
    TRootUP<TTree> tstep_tree_;

    struct TStepData
    {
        int    event_id;
        int    track_id;
        int    action_id;
        int    track_step_count;
        int    particle;          //!< PDG numbering scheme
        double energy_deposition; //!< [MeV]
        double step_length;       //!< [cm]

        // Pre- and post-step specific data
        int                   point_pre_volume_id;
        int                   point_post_volume_id;
        std::array<double, 3> point_pre_dir;
        std::array<double, 3> point_post_dir;
        std::array<double, 3> point_pre_pos;     //!< [cm]
        std::array<double, 3> point_post_pos;    //!< [cm]
        double                point_pre_energy;  //!< [MeV]
        double                point_post_energy; //!< [MeV]
        double                point_pre_time;    //!< [s]
        double                point_post_time;   //!< [s]
    } tstep_;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootStepWriter::RootStepWriter(SPRootFileManager,
                                      SPParticleParams,
                                      StepSelection,
                                      Filters)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void RootStepWriter::execute(StateHostRef const&)
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
