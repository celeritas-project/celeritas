//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/data/StreamStore.hh"
#include "corecel/io/OutputInterface.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/CoreTrackData.hh"

#include "ParticleTallyData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Tally post-step actions for each particle type.
 */
class StepDiagnostic final : public ExplicitActionInterface,
                             public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticle = std::shared_ptr<ParticleParams const>;
    using VecVecCount = std::vector<std::vector<size_type>>;
    //!@}

  public:
    //! Construct with particle data
    StepDiagnostic(ActionId id,
                   SPConstParticle particle,
                   size_type max_bins,
                   size_type num_streams);

    //! Default destructor
    ~StepDiagnostic();

    //!@{
    //! \name ExplicitAction interface
    // Execute action with host data
    void execute(ParamsHostCRef const&, StateHostRef&) const final;
    // Execute action with device data
    void execute(ParamsDeviceCRef const&, StateDeviceRef&) const final;
    //! ID of the action
    ActionId action_id() const final { return id_; }
    //! Short name for the action
    std::string label() const final { return "step-diagnostic"; }
    //! Description of the action for user interaction
    std::string description() const final { return "Step diagnostic"; }
    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::post_post; }
    //!@}

    //!@{
    //! \name Output interface
    //! Category of data to write
    Category category() const final { return Category::result; }
    // Write output to the given JSON object
    void output(JsonPimpl*) const final;
    //!@}

    // Get the diagnostic results accumulated over all streams
    VecVecCount calc_steps() const;

    // Size of diagnostic state data (number of bins times number of particles)
    size_type state_size() const;

    // Reset diagnostic results
    void clear();

  private:
    using StoreT = StreamStore<ParticleTallyParamsData, ParticleTallyStateData>;

    ActionId id_;
    size_type num_streams_;
    mutable StoreT store_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

#if !CELER_USE_DEVICE
inline void
StepDiagnostic::execute(ParamsDeviceCRef const&, StateDeviceRef&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
