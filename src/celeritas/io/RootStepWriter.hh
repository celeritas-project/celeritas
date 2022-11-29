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
 * Write step data to ROOT. `TTree::Fill()` is called for each step and thread
 * id, making each ROOT entry a step. Since the ROOT data is stored in branches
 * with primitive types instead of a full struct, no dictionaries are needed
 * for reading the output file.
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
                            StepSelection     selection,
                            Filters           filters);

    // Set number of entries stored in memory before being flushed to disk
    void set_auto_flush(long num_entries);

    // Process step data on the host and fill step tree
    void execute(StateHostRef const& steps) final;

    // Not implemented
    inline void execute(StateDeviceRef const&) final
    {
        CELER_NOT_IMPLEMENTED("RootStepWriter is host-only");
    }

    // Selection of data to be stored
    StepSelection selection() const final { return selection_; }

    // Selection of filters for the stored data
    Filters filters() const final { return filters_; }

  private:
    void make_tree();
    // Copy pre- and post-step position and direction arrays
    void copy_real3(const Real3& real3, double output[3]);

  private:
    SPRootFileManager  root_manager_;
    SPParticleParams   particles_;
    StepSelection      selection_;
    Filters            filters_;
    TRootUP<TTree>     tstep_tree_;
    mctruth::TStepData tstep_;
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
