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

        // Loop for storing pre- and post-step
        for (auto i : range(StepPoint::size_))
        {
            tstep_.points[(int)i].volume_id = steps.points[i].volume[tid].get();
            tstep_.points[(int)i].energy = steps.points[i].energy[tid].value();
            tstep_.points[(int)i].time   = steps.points[i].time[tid];

            this->copy_real3(steps.points[i].dir[tid],
                             tstep_.points[(int)i].dir);
            this->copy_real3(steps.points[i].pos[tid],
                             tstep_.points[(int)i].pos);
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
    tstep_tree_.reset(new TTree("steps", "steps"));

    // There must be a better way...
#define CREATE_BRANCH_IF_SELECTED(SELECTION, NAME, REF) \
    if (this->SELECTION)                                \
    {                                                   \
        this->tstep_tree_->Branch(NAME, REF);           \
    }

#define CREATE_LEAVES_LIST(SELECTION, NAME, SIZE, LIST)  \
    auto pre  = this->SELECTION.points[StepPoint::pre];  \
    auto post = this->SELECTION.points[StepPoint::post]; \
    if (this->SELECTION.points[StepPoint::pre]           \
        || this->SELECTION.points[StepPoint::post])      \
    {                                                    \
        LIST += NAME + SIZE + "/D:";                     \
    }

#define CREATE_POINTS_BRANCH_IF_SELECTED(SELECTION, NAME, REF, LIST) \
    if (this->SELECTION.points[StepPoint::pre]                       \
        || this->SELECTION.points[StepPoint::post])                  \
    {                                                                \
        this->tstep_tree_->Branch(NAME, REF, LIST.c_str());          \
    }

    // Create branches based on the selection_ booleans
    CREATE_BRANCH_IF_SELECTED(selection_.event, "event", &tstep_.event);
    CREATE_BRANCH_IF_SELECTED(selection_.track_step_count,
                              "track_step_count",
                              &tstep_.track_step_count);
    CREATE_BRANCH_IF_SELECTED(selection_.action, "action", &tstep_.action);

    CREATE_BRANCH_IF_SELECTED(
        selection_.step_length, "step_length", &tstep_.step_length);
    CREATE_BRANCH_IF_SELECTED(
        selection_.particle, "particle", &tstep_.particle);
    CREATE_BRANCH_IF_SELECTED(selection_.energy_deposition,
                              "energy_deposition",
                              &tstep_.energy_deposition);

    const auto pre_step  = selection_.points[StepPoint::pre];
    const auto post_step = selection_.points[StepPoint::post];

    std::string points_leaves;
    std::string points_leaves_size = "";
    if (pre_step && post_step)
    {
        points_leaves_size = "[2]";
    }
    else if (pre_step || post_step)
    {
        points_leaves_size = "[1]";
    }

    if (pre_step.time || post_step.time)
    {
        points_leaves += "time" + points_leaves_size + "/D:";
    }

    if (pre_step.pos || post_step.pos)
    {
        points_leaves += "pos" + points_leaves_size + "[3]/D:";
    }
    if (pre_step.dir || post_step.dir)
    {
        points_leaves += "dir" + points_leaves_size + "[3]/D:";
    }

    if (pre_step.volume || post_step.volume)
    {
        points_leaves += "volume" + points_leaves_size + "/D:";
    }

    if (pre_step.energy || post_step.energy)
    {
        points_leaves += "energy" + points_leaves_size + "/D";
    }

    CREATE_POINTS_BRANCH_IF_SELECTED(
        selection_, "points", &tstep_.points, points_leaves);

#undef CREATE_BRANCH_IF_SELECTED
#undef CREATE_POINTS_BRANCH_IF_SELECTED
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
