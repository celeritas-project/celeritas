//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/FieldDiagnosticExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/math/Atomics.hh"
#include "celeritas/global/CoreTrackView.hh"

#include "../FieldDiagnosticData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Bin the field propagation record of each track slot.
 *
 * This reads the record written by the instrumented propagation kernel
 * earlier in the step, adds it to the accumulated tallies, and invalidates
 * it. Per-track totals are binned and reset when the track is killed.
 */
struct FieldDiagnosticExecutor
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    NativeCRef<FieldDiagnosticParamsData> const params;
    NativeRef<FieldDiagnosticStateData> const state;

  private:
    using SizeId = ItemId<ull_int>;
    using TallyRef
        = Collection<ull_int, Ownership::reference, MemSpace::native>;

    // Add a step record to a row of tallies
    inline CELER_FUNCTION void tally(
        FieldStepRecord const& rec, TallyRef const& items, SizeId start) const;

    // Add to a step iteration record
    inline CELER_FUNCTION void tally_step_iter(
        FieldPropagationCounters const& c) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Bin the propagation record and track totals for a single slot.
 */
CELER_FUNCTION void FieldDiagnosticExecutor::operator()(
    CoreTrackView const& track)
{
    CELER_EXPECT(params);
    CELER_EXPECT(state);

    auto slot = track.track_slot_id();
    FieldTrackTally& totals = state.track[slot];
    ParticleId pid = track.particle().particle_id();

    FieldStepRecord& rec = state.step[slot];
    if (rec.valid)
    {
        rec.valid = false;
        auto const& c = rec.counters;

        // Per-step histograms
        auto add_step_hist = [&](FieldStepHist q, size_type count) {
            atomic_add(&state.step_hist[params.step_hist(pid, q, count)],
                       ull_int{1});
        };
        add_step_hist(FieldStepHist::substeps, c.substeps);
        add_step_hist(FieldStepHist::integrations, c.integrations);
        add_step_hist(FieldStepHist::first_substep_integrations,
                      c.first_substep_integrations);
        add_step_hist(FieldStepHist::intersects, c.intersects);

        // Energy and volume tallies
        this->tally(rec, state.energy, params.energy_tally(rec.energy));
        if (rec.volume && rec.volume.get() < params.num_volumes)
        {
            this->tally(rec, state.volume, params.volume_tally(rec.volume));
        }

        // Ratio of the starting safety to the step length
        if (rec.start_safety >= 0)
        {
            atomic_add(&state.safety[params.safety_hist(
                           pid, rec.start_safety / rec.step)],
                       ull_int{1});
        }

        this->tally_step_iter(c);

        // Accumulate over the track
        ++totals.steps;
        totals.substeps += c.substeps;
        totals.integrations += c.integrations;
        totals.intersects += c.intersects;
    }

    if (track.sim().status() == TrackStatus::killed && totals.steps > 0)
    {
        auto add_track_hist = [&](FieldTrackHist q, size_type count) {
            atomic_add(&state.track_hist[params.track_hist(pid, q, count)],
                       ull_int{1});
        };
        add_track_hist(FieldTrackHist::steps, totals.steps);
        add_track_hist(FieldTrackHist::substeps, totals.substeps);
        add_track_hist(FieldTrackHist::integrations, totals.integrations);
        add_track_hist(FieldTrackHist::intersects, totals.intersects);
        totals = {};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Add a step record to a row of tallies.
 *
 * The row may be in the energy or volume tally collections, which have the
 * same layout.
 */
CELER_FUNCTION void FieldDiagnosticExecutor::tally(
    FieldStepRecord const& rec, TallyRef const& items, SizeId start) const
{
    constexpr size_type num_values = static_cast<size_type>(FieldTally::size_);

    auto const& c = rec.counters;
    bool const stuck = c.stuck > 0;
    ull_int const values[] = {
        1,
        rec.boundary && !stuck,
        rec.looping,
        stuck,
        c.substeps,
        c.accepted,
        c.bumped,
        c.intercepted,
        c.shortened,
        c.integrations,
        c.first_substep_integrations,
        c.quick_advances,
        c.chord_iterations,
        c.accurate_advances,
        c.accurate_iterations,
        c.good_step_rejects,
        c.intersects,
        c.intersect_hits,
        c.skippable_intersects,
        c.safeties,
    };
    static_assert(sizeof(values) / sizeof(values[0]) == num_values);

    for (size_type i = 0; i < num_values; ++i)
    {
        if (values[i] != 0)
        {
            atomic_add(&items[SizeId{start.get() + i}], values[i]);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Add to the record for the current step iteration.
 */
CELER_FUNCTION void FieldDiagnosticExecutor::tally_step_iter(
    FieldPropagationCounters const& c) const
{
    if (state.step_iter >= params.num_step_records)
    {
        return;
    }

    auto at = [this](FieldStepIter q) -> ull_int& {
        return state.step_iters[params.step_iter(state.step_iter, q)];
    };
    atomic_add(&at(FieldStepIter::tracks), ull_int{1});
    atomic_add(&at(FieldStepIter::substeps), ull_int{c.substeps});
    atomic_add(&at(FieldStepIter::integrations), ull_int{c.integrations});
    atomic_add(&at(FieldStepIter::intersects), ull_int{c.intersects});
    atomic_max(&at(FieldStepIter::max_substeps), ull_int{c.substeps});
    atomic_max(&at(FieldStepIter::max_integrations), ull_int{c.integrations});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
