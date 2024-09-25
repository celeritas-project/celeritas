//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SlotDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Span.hh"
#include "corecel/data/AuxInterface.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class AuxStateVec;

//---------------------------------------------------------------------------//
/*!
 * Print diagnostic output about what's in what slots.
 *
 * Currently this only prints the particle ID as a function of track slot,
 * which can later be combined with postprocessing data to print the charge of
 * each particle. We could in the future extend the class to use thread ID
 * and/or write action ID or any other ID/status instead.
 *
 * The filename base is appended with the stream ID, and if it's a directory,
 * that directory must exist. For example, you could pass a filename base of \c
 * slot-diag- to get filenames \c slot-diag-0.jsonl etc.
 *
 * \todo Instead of writing separate files, we should probably use a
 * multi-stream output manager (not yet implemented) to save the result for the
 * end.
 */
class SlotDiagnostic : public AuxParamsInterface, public CoreStepActionInterface
{
  public:
    // Construct with IDs and filename base
    SlotDiagnostic(ActionId action_id, AuxId aux_id, std::string filename_base);

    //!@{
    //! \name Metadata interface
    //! Label for the auxiliary data and action
    std::string_view label() const final { return sad_.label(); }
    // Description of the action
    std::string_view description() const final { return sad_.description(); }
    //!@}

    //!@{
    //! \name Aux params interface
    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }
    // Build state data for a stream
    UPState create_state(MemSpace m, StreamId id, size_type size) const final;
    //!@}

    //!@{
    //! \name Step action interface
    //! Index of this class instance in its registry
    ActionId action_id() const final { return sad_.action_id(); }
    // Execute the action with host data
    void step(CoreParams const& params, CoreStateHost& state) const final;
    // Execute the action with device data
    void step(CoreParams const& params, CoreStateDevice& state) const final;
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
