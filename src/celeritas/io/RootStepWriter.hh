//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootStepWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Assert.hh"
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

    // Process step data on the host and fill step tree
    void execute(StateHostRef const& steps) final;

    // Not implemented
    void execute(StateDeviceRef const&) final
    {
        CELER_NOT_IMPLEMENTED("RootStepWriter is host-only");
    }

    // Selection of data to be stored
    StepSelection selection() const final { return selection_; }

  private:
    void make_tree();
    // Copy pre- and post-step position and direction arrays
    void copy_real3(const Real3& real3, double output[3]);

  private:
    SPRootFileManager  root_manager_;
    SPParticleParams   particles_;
    StepSelection      selection_;
    TRootUP<TTree>     tstep_tree_;
    TRootUP<TBranch>   tstep_branch_;
    mctruth::TStepData tstep_;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
RootStepWriter::RootStepWriter(SPRootFileManager,
                               SPParticleParams,
                               StepSelection)
{
    CELER_NOT_CONFIGURED("ROOT");
}

void RootStepWriter::execute(StateHostRef const&)
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
