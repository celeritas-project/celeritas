//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SlotDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/cont/Span.hh"
#include "corecel/data/AuxInterface.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class AuxStateVec;
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Print diagnostic output about what's in what slots.
 *
 * Currently this only prints the particle ID as a function of track slot,
 * which can later be combined with postprocessing data to print the charge of
 * each particle. We could in the future extend the class to use thread ID
 * and/or write action ID or any other ID/status instead. Special IDs are:
 * - \c -1 : track slot is inactive
 * - \c -2 : track has been flagged as an error
 *
 * A "JSON lines" file (one line per step) is opened for each stream, and a
 * separate file is opened once during construction to write appropriate
 * metadata.
 *
 * The filename base is appended with the stream ID or \c metadata. If
 * the filename is a directory, that directory must already exist.
 * For example, you could pass a filename base of \c
 * slot-diag- to get filenames \c slot-diag-metadata.json, \c
 * slot-diag-0.jsonl, etc.
 *
 * \todo Instead of writing separate files, we should probably use a
 * multi-stream output manager (not yet implemented) to save the result for the
 * end.
 *
 * \note To plot the resulting files, see \c
 * scripts/user/plot-slot-diagnostic.py
 */
class SlotDiagnostic final : public CoreStepActionInterface,
                             public AuxParamsInterface
{
  public:
    // Construct and add to core params
    static std::shared_ptr<SlotDiagnostic>
    make_and_insert(CoreParams const& core, std::string filename_base);

    // Construct with IDs and filename base
    SlotDiagnostic(ActionId action_id,
                   AuxId aux_id,
                   std::string filename_base,
                   size_type num_stream,
                   std::shared_ptr<ParticleParams const> particle);

    //!@{
    //! \name Metadata interface

    //! Label for the auxiliary data and action
    std::string_view label() const final { return sad_.label(); }
    // Description of the action
    std::string_view description() const final { return sad_.description(); }
    //!@}

    //!@{
    //! \name Step action interface

    //! Index of this class instance in the action registry
    ActionId action_id() const final { return sad_.action_id(); }
    //! Ordering of the action inside the step
    StepActionOrder order() const final { return StepActionOrder::user_post; }
    // Execute the action with host data
    void step(CoreParams const& params, CoreStateHost& state) const final;
    // Execute the action with device data
    void step(CoreParams const& params, CoreStateDevice& state) const final;
    //!@}

    //!@{
    //! \name Aux params interface

    //! Index of this class instance in the aux registry
    AuxId aux_id() const final { return aux_id_; }
    // Build state data for a stream
    UPState create_state(MemSpace m, StreamId id, size_type size) const final;
    //!@}

  private:
    struct State;

    //// DATA ////

    StaticActionData sad_;
    AuxId aux_id_;
    std::string filename_base_;

    //// HELPER FUNCTIONS ////

    Span<int> get_host_buffer(AuxStateVec&) const;
    void write_buffer(AuxStateVec&) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
