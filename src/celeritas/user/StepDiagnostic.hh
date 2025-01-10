//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
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
 *
 * This adds an \c step-diagnostic entry to the \c result category of the
 * main Celeritas output that bins the total number of steps taken by a track,
 * grouped by particle type. The result is an integral over all events.
 */
class StepDiagnostic final : public CoreStepActionInterface,
                             public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticle = std::shared_ptr<ParticleParams const>;
    using VecVecCount = std::vector<std::vector<size_type>>;
    //!@}

  public:
    // Construct and add to core params
    static std::shared_ptr<StepDiagnostic>
    make_and_insert(CoreParams const& core, size_type max_bins);

    //! Construct with particle data
    StepDiagnostic(ActionId id,
                   SPConstParticle particle,
                   size_type max_bins,
                   size_type num_streams);

    //! Default destructor
    ~StepDiagnostic() final;

    //!@{
    //! \name ExplicitAction interface

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
    //! ID of the action
    ActionId action_id() const final { return id_; }
    //! Short name for the action
    std::string_view label() const final { return "step-diagnostic"; }
    // Description of the action for user interaction
    std::string_view description() const final;
    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::user_post; }
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
}  // namespace celeritas
