//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/RootStepWriter.cc
//---------------------------------------------------------------------------//
#include "RootStepWriter.hh"

#include <algorithm>
#include <cstring>
#include <TBranch.h>
#include <TFile.h>
#include <TTree.h>

#include "corecel/Assert.hh"
#include "celeritas/ext/RootFileManager.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Copy StepStateData collection values to TStepData.
 */
template<class T1, class T2>
void copy_if_selected(const T1& src, T2& dst)
{
    dst = src;
}

//---------------------------------------------------------------------------//
/*!
 * Copy StepPointStateData values to TStepPoint arrays.
 */
void copy_if_selected(Real3 const& src, std::array<real_type, 3>& dst)
{
    std::copy(src.begin(), src.end(), dst.begin());
}

//---------------------------------------------------------------------------//
//!@{
//! SimpleRootFilter helpers: true if unspecified or matching.
bool srf_match(size_type step_attr_id, size_type filter_id)
{
    return filter_id == SimpleRootFilterInput::unspecified
           || step_attr_id == filter_id;
}

bool srf_match(size_type step_trk_id, std::vector<size_type> const& vec_trk_id)
{
    if (vec_trk_id.empty())
    {
        // No track ID filter specified
        return true;
    }
    else
    {
        auto iter
            = std::find(vec_trk_id.begin(), vec_trk_id.end(), step_trk_id);
        // True if step track ID is in the list of IDs
        return iter != vec_trk_id.end();
    }
}
//!@}
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct writer with user-defined data filtering.
 */
RootStepWriter::RootStepWriter(SPRootFileManager root_manager,
                               SPParticleParams particles,
                               StepSelection selection,
                               WriteFilter filter)
    : root_manager_(std::move(root_manager))
    , particles_(std::move(particles))
    , selection_(selection)
    , filter_(std::move(filter))
{
    CELER_EXPECT(root_manager_);

    if (!filter_)
    {
        // Write all data by default
        filter_ = [](RootStepWriter::TStepData const&) { return true; };
    }

    this->make_tree();
}

//---------------------------------------------------------------------------//
/*!
 * Construct writer without data filtering.
 */
RootStepWriter::RootStepWriter(SPRootFileManager root_manager,
                               SPParticleParams particle_params,
                               StepSelection selection)
    : RootStepWriter(std::move(root_manager),
                     std::move(particle_params),
                     std::move(selection),
                     nullptr)
{
}

//---------------------------------------------------------------------------//
/*!
 * Set the number of entries before flushing to disk.
 *
 * This sets the number of steps stored in memory before ROOT flushes the data
 * to disk. Default is ~32MB of compressed data.
 *
 * See `SetAutoFlush` in ROOT TTree Class reference for details:
 * https://root.cern.ch/doc/master/classTTree.html
 */
void RootStepWriter::set_auto_flush(long num_entries)
{
    CELER_EXPECT(root_manager_);
    CELER_EXPECT(tstep_tree_);
    tstep_tree_->SetAutoFlush(num_entries);
}

//---------------------------------------------------------------------------//
/*!
 * Collect step data and fill the ROOT TTree for all active threads.
 */
void RootStepWriter::process_steps(HostStepState state)
{
#define RSW_STORE(ATTR, GETTER)                                               \
    do                                                                        \
    {                                                                         \
        if (selection_.ATTR)                                                  \
        {                                                                     \
            copy_if_selected(state.steps.data.ATTR[tid] GETTER, tstep_.ATTR); \
        }                                                                     \
    } while (0)

    CELER_EXPECT(state.steps);
    if (state.stream_id != StreamId{0})
    {
        CELER_NOT_IMPLEMENTED("thread-safe ROOT output");
    }
    tstep_ = TStepData();

    // Loop over track slots and fill TTree
    for (auto const tid : range(TrackSlotId{state.steps.size()}))
    {
        if (!state.steps.data.track_id[tid])
        {
            // Track id not found; skip inactive track slot
            continue;
        }

        // Track id is always set
        tstep_.track_id = state.steps.data.track_id[tid].unchecked_get();

        RSW_STORE(event_id, .get());
        RSW_STORE(parent_id, .unchecked_get());
        RSW_STORE(action_id, .get());
        RSW_STORE(energy_deposition, .value());
        RSW_STORE(step_length, /* no getter */);
        RSW_STORE(track_step_count, /* no getter */);
        if (selection_.particle)
        {
            copy_if_selected(
                particles_->id_to_pdg(state.steps.data.particle[tid]).get(),
                tstep_.particle);
        }

        for (auto const sp : range(StepPoint::size_))
        {
            RSW_STORE(points[sp].volume_id, .unchecked_get());
            RSW_STORE(points[sp].energy, .value());
            RSW_STORE(points[sp].time, /* no getter */);
            RSW_STORE(points[sp].dir, /* no getter */);
            RSW_STORE(points[sp].pos, /* no getter */);
        }

        if (filter_(tstep_))
        {
            tstep_tree_->Fill();
        }
    }

#undef RSW_STORE
}

//---------------------------------------------------------------------------//
/*!
 * Create steps tree.
 *
 * In order to have the option to individually select any
 * member of `StepStateData` (StepData.hh) to be stored into the ROOT file
 * *and* not need any dictionary for ROOT I/O, we cannot store an MC truth step
 * object. Therefore, the data is flattened so that each member of `TStepData`
 * is an individual branch that stores primitive types and is created based on
 * the `StepSelection` booleans.
 */
void RootStepWriter::make_tree()
{
#define RSW_CREATE_BRANCH(ATTR, BRANCH_NAME)                      \
    do                                                            \
    {                                                             \
        if (this->selection_.ATTR)                                \
        {                                                         \
            this->tstep_tree_->Branch(BRANCH_NAME, &tstep_.ATTR); \
        }                                                         \
    } while (0)

    tstep_tree_ = root_manager_->make_tree("steps", "steps");

    tstep_tree_->Branch("track_id", &tstep_.track_id);  // Always on
    RSW_CREATE_BRANCH(event_id, "event_id");
    RSW_CREATE_BRANCH(parent_id, "parent_id");
    RSW_CREATE_BRANCH(track_step_count, "track_step_count");
    RSW_CREATE_BRANCH(action_id, "action_id");
    RSW_CREATE_BRANCH(step_length, "step_length");
    RSW_CREATE_BRANCH(particle, "particle");
    RSW_CREATE_BRANCH(energy_deposition, "energy_deposition");
    // Pre-step
    RSW_CREATE_BRANCH(points[StepPoint::pre].volume_id, "pre_volume_id");
    RSW_CREATE_BRANCH(points[StepPoint::pre].dir, "pre_dir");
    RSW_CREATE_BRANCH(points[StepPoint::pre].pos, "pre_pos");
    RSW_CREATE_BRANCH(points[StepPoint::pre].energy, "pre_energy");
    RSW_CREATE_BRANCH(points[StepPoint::pre].time, "pre_time");
    // Post-step
    RSW_CREATE_BRANCH(points[StepPoint::post].volume_id, "post_volume_id");
    RSW_CREATE_BRANCH(points[StepPoint::post].dir, "post_dir");
    RSW_CREATE_BRANCH(points[StepPoint::post].pos, "post_pos");
    RSW_CREATE_BRANCH(points[StepPoint::post].energy, "post_energy");
    RSW_CREATE_BRANCH(points[StepPoint::post].time, "post_time");

#undef RSW_CREATE_BRANCH
}

//---------------------------------------------------------------------------//
RootStepWriter::WriteFilter make_write_filter(SimpleRootFilterInput const& inp)
{
    if (!inp)
    {
        // No filtering
        return nullptr;
    }

    return [inp](RootStepWriter::TStepData const& step) {
        if (inp.action_id != SimpleRootFilterInput::unspecified)
        {
            return step.action_id == inp.action_id;
        }
        return (srf_match(step.event_id, inp.event_id)
                && srf_match(step.track_id, inp.track_id)
                && srf_match(step.parent_id, inp.parent_id));
    };
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
