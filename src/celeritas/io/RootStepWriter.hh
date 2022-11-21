//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootStepWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/io/MCTruthData.hh"
#include "celeritas/io/RootFileManager.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/user/StepInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 */
class RootStepWriter final : public StepInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPRootFileManager = std::shared_ptr<RootFileManager>;
    using SPParticleParams  = std::shared_ptr<const ParticleParams>;
    template<class T>
    using TRootUP = detail::TRootUniquePtr<T>;
    //!@}

    // Construct with RootFileManager and ParticleParams
    explicit RootStepWriter(SPRootFileManager root_manager,
                            SPParticleParams  particle_params,
                            StepSelection     selection);

    // Set number of entries stored in memory before being flushed to disk
    void set_auto_flush(long num_entries);

    // Process step data and fill step tree
    void execute(StateHostRef const& steps) final;
    void execute(StateDeviceRef const& steps) final;

    // Selection of data to be stored
    StepSelection selection() const final { return selection_; }

  private:
    SPRootFileManager  root_manager_;
    SPParticleParams   particles_;
    StepSelection      selection_;
    TRootUP<TTree>     tstep_tree_;
    TRootUP<TBranch>   tstep_branch_;
    mctruth::TStepData tstep_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
