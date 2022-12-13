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

    // Construct with RootFileManager, ParticleParams, and data selection
    RootStepWriter(SPRootFileManager root_manager,
                   SPParticleParams  particle_params,
                   StepSelection     selection);

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

  private:
    //// TYPES ////

    // Truth step point data; Naming convention *must* match StepPointStateData
    struct TStepPoint
    {
        int                   volume_id;
        double                energy; //!< [MeV]
        double                time;   //!< [s]
        std::array<double, 3> pos;    //!< [cm]
        std::array<double, 3> dir;
    };

    // Full truth step data; Naming convention *must* match StepStateData
    struct TStepData
    {
        int                              event_id;
        int                              track_id;
        int                              action_id;
        int                              track_step_count;
        int                              particle;          //!< PDG number
        double                           energy_deposition; //!< [MeV]
        double                           step_length;       //!< [cm]
        EnumArray<StepPoint, TStepPoint> points;
    };

    //// DATA ////

    SPRootFileManager            root_manager_;
    SPParticleParams             particles_;
    StepSelection                selection_;
    detail::RootUniquePtr<TTree> tstep_tree_;
    // Members of tstep_ are used as references for the step TTree branches
    TStepData tstep_;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootStepWriter::RootStepWriter(SPRootFileManager,
                                      SPParticleParams,
                                      StepSelection)
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
