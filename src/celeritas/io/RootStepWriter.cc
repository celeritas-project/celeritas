//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootStepWriter.cc
//---------------------------------------------------------------------------//
#include "RootStepWriter.hh"

#include <algorithm>
#include <tuple>
#include <TBranch.h>
#include <TFile.h>
#include <TTree.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct writer with RootFileManager, ParticleParams (to convert particle
 * id to pdg), and the selection of data to be tallied.
 */
RootStepWriter::RootStepWriter(SPRootFileManager io_manager,
                               SPParticleParams  particle_params,
                               StepSelection     selection)
    : StepInterface()
    , root_manager_(io_manager)
    , particles_(particle_params)
    , selection_(selection)
{
    CELER_EXPECT(root_manager_);
    this->make_tree();
}

//---------------------------------------------------------------------------//
/*!
 * Set the number of entries (i.e. number of steps) stored in memory before
 * ROOT flushes the data to disk. Default is ~32MB of compressed data.
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
 * Collect step data from each track on each thread id.
 */
void RootStepWriter::execute(StateHostRef const& steps)
{
#define IF_SELECTED(ATTR, VAL) \
    do                         \
    {                          \
        if (selection_.ATTR)   \
        {                      \
            tstep_.ATTR = VAL; \
        }                      \
    } while (0)

#define IF_POINT_SELECTED(ATTR, VAL) \
    do                               \
    {                                \
        if (selection_point.ATTR)    \
        {                            \
            tpoint.ATTR = VAL;       \
        }                            \
    } while (0)

#define IF_POINT_REAL3_SELECTED(ATTR, TPOINT_ATTR)          \
    do                                                      \
    {                                                       \
        if (selection_point.ATTR)                           \
        {                                                   \
            this->copy_real3(point.ATTR[tid], TPOINT_ATTR); \
        }                                                   \
    } while (0)

    // TODO: add selection_ options for storing data
    CELER_EXPECT(steps);
    tstep_ = mctruth::TStepData();

    // Loop over thread ids and fill TTree
    for (const auto tid : range(ThreadId{steps.size()}))
    {
        if (!steps.track_id[tid])
        {
            // Track id not found; skip inactive track slot
            continue;
        }

        // Track id is always set
        tstep_.track_id = steps.track_id[tid].unchecked_get();

        IF_SELECTED(event_id, steps.event_id[tid].get());
        IF_SELECTED(action_id, steps.action_id[tid].get());
        IF_SELECTED(particle, particles_->id_to_pdg(steps.particle[tid]).get());
        IF_SELECTED(energy_deposition, steps.energy_deposition[tid].value());
        IF_SELECTED(step_length, steps.step_length[tid]);
        IF_SELECTED(track_step_count, steps.track_step_count[tid]);

        // Store pre- and post-step
        for (auto i : range(StepPoint::size_))
        {
            const auto& selection_point = selection_.points[i];
            const auto& point           = steps.points[i];
            auto&       tpoint          = tstep_.points[(int)i];

            IF_POINT_SELECTED(volume_id, point.volume_id[tid].get());
            IF_POINT_SELECTED(energy, point.energy[tid].value());
            IF_POINT_SELECTED(time, point.time[tid]);
            IF_POINT_REAL3_SELECTED(pos, tpoint.pos);
            IF_POINT_REAL3_SELECTED(dir, tpoint.dir);
        }

        tstep_tree_->Fill();
    }

#undef IF_SELECTED
#undef IF_POINT_SELECTED
#undef IF_POINT_REAL3_SELECTED
}

//---------------------------------------------------------------------------//
/*!
 * Set up steps tree. In order to have the option to individually select any
 * member of `StepStateData` (defined in StepData.hh) to be stored into the
 * ROOT file, we cannot store an MC truth step object. This is accomplished by
 * "flattening" the data so that each member of `TStepData` (MCTruthData.hh) is
 * an individual branch that can be created based on the `StepSelection`
 * booleans.
 *
 * To simplify the process of moving from Collection to a ROOT branch
 * `TStepData` members *must* have the *exact* same name as the Collection
 * members of `StepStateData`. This allows the use of C++ macros to
 * automatically create ROOT branches with the same names as the struct
 * elements.
 */
void RootStepWriter::make_tree()
{
#define CREATE_BRANCH_IF_SELECTED(ATTR)                     \
    do                                                      \
    {                                                       \
        if (this->selection_.ATTR)                          \
        {                                                   \
            this->tstep_tree_->Branch(#ATTR, &tstep_.ATTR); \
        }                                                   \
    } while (0)

#define CREATE_POINT_BRANCH_IF_SELECTED(ATTR, NAME, REF, TYPE)           \
    do                                                                   \
    {                                                                    \
        if (this->selection_.ATTR)                                       \
        {                                                                \
            std::string leaf_def(NAME);                                  \
            std::string str_attr(#ATTR);                                 \
            const auto  last_p = str_attr.find_last_of(".");             \
            if (str_attr.find("pos", last_p, 3) != std::string::npos     \
                || str_attr.find("dir", last_p, 3) != std::string::npos) \
            {                                                            \
                leaf_def += "[3]";                                       \
            }                                                            \
            leaf_def += "/" + std::string(TYPE);                         \
            this->tstep_tree_->Branch(NAME, &REF, leaf_def.c_str());     \
        }                                                                \
    } while (0)

    tstep_tree_.reset(new TTree("steps", "steps"));
    tstep_tree_->Branch("track_id", &tstep_.track_id); // Always on

    // Step selection data
    CREATE_BRANCH_IF_SELECTED(event_id);
    CREATE_BRANCH_IF_SELECTED(track_step_count);
    CREATE_BRANCH_IF_SELECTED(action_id);
    CREATE_BRANCH_IF_SELECTED(step_length);
    CREATE_BRANCH_IF_SELECTED(particle);
    CREATE_BRANCH_IF_SELECTED(energy_deposition);

    // Step point selection data //
    // Pre-step
    {
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::pre].volume_id,
                                        "points.pre.volume_id",
                                        tstep_.points[0].volume_id,
                                        "I");
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::pre].dir,
                                        "points.pre.dir",
                                        tstep_.points[0].dir,
                                        "D");
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::pre].pos,
                                        "points.pre.pos",
                                        tstep_.points[0].pos,
                                        "D");
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::pre].energy,
                                        "points.pre.energy",
                                        tstep_.points[0].energy,
                                        "D");
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::pre].time,
                                        "points.pre.time",
                                        tstep_.points[0].time,
                                        "D");
    }

    // Post-step
    {
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::post].volume_id,
                                        "points.post.volume_id",
                                        tstep_.points[1].volume_id,
                                        "I");
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::post].dir,
                                        "points.post.dir",
                                        tstep_.points[1].dir,
                                        "D");
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::post].pos,
                                        "points.post.pos",
                                        tstep_.points[1].pos,
                                        "D");
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::post].energy,
                                        "points.post.energy",
                                        tstep_.points[1].energy,
                                        "D");
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::post].time,
                                        "points.post.time",
                                        tstep_.points[1].time,
                                        "D");
    }

#undef CREATE_BRANCH_IF_SELECTED
#undef CREATE_POINT_BRANCH_IF_SELECTED
}

//---------------------------------------------------------------------------//
/*!
 * Copy pre- and post-step position and direction arrays.
 */
void RootStepWriter::copy_real3(const Real3& input, double output[3])
{
    std::copy(input.begin(), input.end(), output);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
