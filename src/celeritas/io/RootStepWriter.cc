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

#include "corecel/io/Logger.hh"

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
 * See ROOT TTree Class reference for details:
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
    // TODO: add selection_ options for storing data
    CELER_EXPECT(steps);
    tstep_ = mctruth::TStepData();

    // Loop over thread ids and fill TTree
    for (const auto tid : range(ThreadId{steps.size()}))
    {
        if (!steps.track[tid])
        {
            // Track id not found; skip inactive track slot
            continue;
        }

        tstep_.event    = steps.event[tid].get();
        tstep_.track    = steps.track[tid].unchecked_get();
        tstep_.action   = steps.action[tid].get();
        tstep_.particle = particles_->id_to_pdg(steps.particle[tid]).get();
        tstep_.energy_deposition = steps.energy_deposition[tid].value();
        tstep_.step_length       = steps.step_length[tid];
        tstep_.track_step_count  = steps.track_step_count[tid];

        if (!selection_.points[StepPoint::pre]
            && !selection_.points[StepPoint::post])
        {
            // Do not store step point data
            continue;
        }

        // Store pre- and post-step
        for (auto i : range(StepPoint::size_))
        {
            tstep_.points[(int)i].volume = steps.points[i].volume[tid].get();
            tstep_.points[(int)i].energy = steps.points[i].energy[tid].value();
            tstep_.points[(int)i].time   = steps.points[i].time[tid];

            if (selection_.points[i].dir)
            {
                this->copy_real3(steps.points[i].dir[tid],
                                 tstep_.points[(int)i].dir);
            }
            if (selection_.points[i].pos)
            {
                this->copy_real3(steps.points[i].pos[tid],
                                 tstep_.points[(int)i].pos);
            }
        }

        tstep_tree_->Fill();
    }
}

//---------------------------------------------------------------------------//
/*!
 * TBD
 */
void RootStepWriter::make_tree()
{
#define CREATE_BRANCH_IF_SELECTED(ATTR, REF)        \
    do                                              \
    {                                               \
        if (this->selection_.ATTR)                  \
        {                                           \
            this->tstep_tree_->Branch(#ATTR, &REF); \
        }                                           \
    } while (0)

#define CREATE_POINT_BRANCH_IF_SELECTED(ATTR, NAME, REF)                 \
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
            leaf_def += "/D";                                            \
            this->tstep_tree_->Branch(NAME, &REF, leaf_def.c_str());     \
        }                                                                \
    } while (0)

    tstep_tree_.reset(new TTree("steps", "steps"));

    // Step selection data
    {
        CREATE_BRANCH_IF_SELECTED(event, tstep_.event);
        CREATE_BRANCH_IF_SELECTED(track_step_count, tstep_.track_step_count);
        CREATE_BRANCH_IF_SELECTED(action, tstep_.action);
        CREATE_BRANCH_IF_SELECTED(step_length, tstep_.step_length);
        CREATE_BRANCH_IF_SELECTED(particle, tstep_.particle);
        CREATE_BRANCH_IF_SELECTED(energy_deposition, tstep_.energy_deposition);
    }

    // Step point selection data
    {
        // Pre-step
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::pre].volume,
                                        "points.pre.volume",
                                        tstep_.points[0].volume);
        CREATE_POINT_BRANCH_IF_SELECTED(
            points[StepPoint::pre].dir, "points.pre.dir", tstep_.points[0].dir);
        CREATE_POINT_BRANCH_IF_SELECTED(
            points[StepPoint::pre].pos, "points.pre.pos", tstep_.points[0].pos);
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::pre].energy,
                                        "points.pre.energy",
                                        tstep_.points[0].energy);
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::pre].time,
                                        "points.pre.time",
                                        tstep_.points[0].time);
        // Post-step
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::post].volume,
                                        "points.post.volume",
                                        tstep_.points[1].volume);
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::post].dir,
                                        "points.post.dir",
                                        tstep_.points[1].dir);
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::post].pos,
                                        "points.post.pos",
                                        tstep_.points[1].pos);
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::post].energy,
                                        "points.post.energy",
                                        tstep_.points[1].energy);
        CREATE_POINT_BRANCH_IF_SELECTED(points[StepPoint::post].time,
                                        "points.post.time",
                                        tstep_.points[1].time);
    }

#undef CREATE_BRANCH_IF_SELECTED
#undef CREATE_POINT_BRANCH_IF_SELECTED
} // namespace celeritas

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
