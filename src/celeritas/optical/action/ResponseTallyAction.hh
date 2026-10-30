//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/ResponseTallyAction.hh
//! \sa ResponseTally.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/data/StreamStore.hh"
#include "corecel/io/OutputInterface.hh"
#include "celeritas/optical/ResponseTallyData.hh"

#include "ActionInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Tally optical photon detector hits on a grid of detector, x, y, z, and time.
 *
 * After the detector action has killed photons that reached a sensitive
 * detector, this action completes each hit's compressed bin ID (see \c
 * ResponseTallyView) and increments the per-stream hit count. The spatial
 * (x, y, z) part of the bin is that of the photon's \em starting point,
 * which is saved by the companion \c ResponseTallyStartAction when the
 * track is created; the detector and time are those of the hit. Hits that lie
 * outside the grids are discarded.
 *
 * At the end of the run, \c calc_counts reduces the hit counts over all
 * streams (host and device), and \c calc_response normalizes them by the
 * total number of photons created, which is supplied externally since it need
 * not be stored on device. The \c output method writes the grid definition
 * and raw counts as JSON under the \c result category.
 *
 * The flattened bin index is in C order over <code>{detector, x, y, z,
 * time}</code>, with time changing fastest.
 */
class ResponseTallyAction final : public OpticalStepActionInterface,
                                  public OutputInterface
{
  public:
    //!@{
    //! \name Type aliases
    using StoreT = StreamStore<ResponseTallyParamsData, ResponseTallyStateData>;
    using HostParams = HostVal<ResponseTallyParamsData>;
    //!@}

  public:
    // Construct and add to core params
    static std::shared_ptr<ResponseTallyAction> make_and_insert(
        CoreParams const& core, HostParams&& params);

    // Construct with ID, tally definition, and number of streams
    ResponseTallyAction(
        ActionId id, HostParams&& params, size_type num_streams);

    //!@{
    //! \name StepAction interface

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
    //! ID of the action
    ActionId action_id() const final { return id_; }
    //! Short name for the action
    std::string_view label() const final { return "optical-response-tally"; }
    // Description of the action for user interaction
    std::string_view description() const final;
    //! Must run after the detector action
    StepActionOrder order() const final { return StepActionOrder::user_post; }
    //!@}

    //!@{
    //! \name Output interface

    //! Category of data to write
    Category category() const final { return Category::result; }
    // Write output to the given JSON object
    void output(JsonPimpl*) const final;
    //!@}

    // Hit counts accumulated over all streams
    std::vector<size_type> calc_counts() const;

    // Hit counts normalized by the number of photons created
    std::vector<double> calc_response(size_type num_photons) const;

    // Reset accumulated counts
    void clear();

    //! Access the tally definition
    HostCRef<ResponseTallyParamsData> const& params() const
    {
        return store_.params<MemSpace::host>();
    }

    //! Access the storage
    StoreT& store() const { return store_; }

  private:
    ActionId id_;
    mutable StoreT store_;
};

//---------------------------------------------------------------------------//
/*!
 * Save the spatial bin of newly created photons for the response tally.
 *
 * This runs after track initialization and before the first step, so the
 * photon is still at its emission point (for wavelength-shifted secondaries,
 * the point where the secondary is emitted). It writes into the state of the
 * \c ResponseTallyAction that owns the tally data.
 */
class ResponseTallyStartAction final : public OpticalStepActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPTally = std::shared_ptr<ResponseTallyAction>;
    //!@}

  public:
    // Construct with ID and the tally to fill
    ResponseTallyStartAction(ActionId id, SPTally tally);

    //!@{
    //! \name StepAction interface

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
    //! ID of the action
    ActionId action_id() const final { return id_; }
    //! Short name for the action
    std::string_view label() const final
    {
        return "optical-response-tally-start";
    }
    // Description of the action for user interaction
    std::string_view description() const final;
    //! Runs after initialization, before the first step
    StepActionOrder order() const final { return StepActionOrder::user_start; }
    //!@}

  private:
    ActionId id_;
    SPTally tally_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
