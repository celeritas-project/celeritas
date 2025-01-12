//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/RootStepWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <array>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "celeritas/ext/RootUniquePtr.hh"

#include "RootStepWriterInput.hh"
#include "StepInterface.hh"

class TTree;

namespace celeritas
{
//---------------------------------------------------------------------------//
class ParticleParams;
class RootFileManager;

//---------------------------------------------------------------------------//
/*!
 * Write "MC truth" data to ROOT at every step.
 *
 * `TTree::Fill()` is called for each step and thread id, making each ROOT
 * entry a step. Since the ROOT data is stored in branches with primitive types
 * instead of a full struct, no dictionaries are needed for reading the output
 * file.
 *
 * The step data that is written to the ROOT file can be filtered by providing
 * a user-defined `WriteFilter` function.
 */
class RootStepWriter final : public StepInterface
{
  public:
    // Unspecified step attribute data value
    static constexpr size_type unspecified{static_cast<size_type>(-1)};

    //! Truth step point data; Naming convention must match StepPointStateData
    struct TStepPoint
    {
        size_type volume_id = unspecified;
        real_type energy = 0;  //!< [MeV]
        real_type time = 0;  //!< [time]
        std::array<real_type, 3> pos{0, 0, 0};  //!< [len]
        std::array<real_type, 3> dir{0, 0, 0};
    };

    //! Truth step data; Naming convention must match StepStateData
    struct TStepData
    {
        size_type event_id = unspecified;
        size_type track_id = unspecified;
        size_type parent_id = unspecified;
        size_type action_id = unspecified;
        size_type track_step_count = unspecified;
        int particle = 0;  //!< PDG number
        real_type energy_deposition = 0;  //!< [MeV]
        real_type step_length = 0;  //!< [len]
        EnumArray<StepPoint, TStepPoint> points;
    };

  public:
    //!@{
    //! \name Type aliases
    using SPRootFileManager = std::shared_ptr<RootFileManager>;
    using SPParticleParams = std::shared_ptr<ParticleParams const>;
    using WriteFilter = std::function<bool(TStepData const&)>;
    //!@}

    // Construct with step data writer filter
    RootStepWriter(SPRootFileManager root_manager,
                   SPParticleParams particle_params,
                   StepSelection selection,
                   WriteFilter filter);

    // Construct and store all step data
    RootStepWriter(SPRootFileManager root_manager,
                   SPParticleParams particle_params,
                   StepSelection selection);

    // Set number of entries stored in memory before being flushed to disk
    void set_auto_flush(long num_entries);

    // Process step data on the host and fill step tree
    void process_steps(HostStepState) final;

    // Device execution is not currently implemented
    void process_steps(DeviceStepState) final
    {
        CELER_NOT_IMPLEMENTED("RootStepWriter with device data");
    }

    // Selection of data to be stored
    StepSelection selection() const final { return selection_; }

    // No detector filtering selection is implemented
    Filters filters() const final { return {}; }

  private:
    // Create steps tree based on selection_ booleans
    void make_tree();

  private:
    SPRootFileManager root_manager_;
    SPParticleParams particles_;
    StepSelection selection_;
    UPRootTreeWritable tstep_tree_;
    TStepData tstep_;  // Members are used as refs of the TTree branches
    std::function<bool(TStepData const&)> filter_;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Create a write filter for some simple IDs
RootStepWriter::WriteFilter make_write_filter(SimpleRootFilterInput const&);

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootStepWriter::RootStepWriter(SPRootFileManager,
                                      SPParticleParams,
                                      StepSelection,
                                      WriteFilter)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void RootStepWriter::process_steps(HostStepState)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline RootStepWriter::WriteFilter
make_write_filter(SimpleRootFilterInput const&)
{
    return nullptr;
}

#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
