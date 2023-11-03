//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DiagnosticTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Types.hh"

#include "DiagnosticData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * User diagnostic properties for a single track.
 *
 * Each field will be present *only* if a user diagnostic that requires it is
 * enabled.
 */
class DiagnosticTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using DiagnosticParamsRef = NativeCRef<DiagnosticParamsData>;
    using DiagnosticStateRef = NativeRef<DiagnosticStateData>;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct with view to state and persistent data
    inline CELER_FUNCTION DiagnosticTrackView(DiagnosticParamsRef const& params,
                                              DiagnosticStateRef const& states,
                                              TrackSlotId tid);

    // Store the number of field substeps
    inline CELER_FUNCTION void num_field_substeps(size_type);

    // Set the pre-step energy
    inline CELER_FUNCTION void pre_step_energy(Energy);

    // Number of field substeps in this step
    CELER_FORCEINLINE_FUNCTION size_type num_field_substeps() const;

    // Access the pre-step energy
    CELER_FORCEINLINE_FUNCTION Energy pre_step_energy() const;

  private:
    DiagnosticParamsRef const& params_;
    DiagnosticStateRef const& states_;
    const TrackSlotId track_slot_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent and local data.
 */
CELER_FUNCTION
DiagnosticTrackView::DiagnosticTrackView(DiagnosticParamsRef const& params,
                                         DiagnosticStateRef const& states,
                                         TrackSlotId tid)
    : params_(params), states_(states), track_slot_(tid)
{
    CELER_EXPECT(!params_.field_diagnostic || track_slot_ < states_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Store the number of field substeps.
 */
CELER_FUNCTION void
DiagnosticTrackView::num_field_substeps(size_type num_substeps)
{
    if (params_.field_diagnostic)
    {
        states_.num_field_substeps[track_slot_] = num_substeps;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Save the pre-step energy.
 */
CELER_FUNCTION void DiagnosticTrackView::pre_step_energy(Energy energy)
{
    if (params_.field_diagnostic)
    {
        states_.pre_step_energy[track_slot_] = energy;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Number of field substeps in this step.
 */
CELER_FUNCTION size_type DiagnosticTrackView::num_field_substeps() const
{
    if (params_.field_diagnostic)
    {
        return states_.num_field_substeps[track_slot_];
    }
    return 0;
}

//---------------------------------------------------------------------------//
/*!
 * Access the pre-step energy.
 */
CELER_FUNCTION auto DiagnosticTrackView::pre_step_energy() const -> Energy
{
    if (params_.field_diagnostic)
    {
        return states_.pre_step_energy[track_slot_];
    }
    return zero_quantity();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
