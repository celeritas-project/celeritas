//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsParams.cc
//---------------------------------------------------------------------------//
#include "PhysicsParams.hh"

#include <algorithm>
#include "base/Range.hh"
#include "base/VectorUtils.hh"
#include "ParticleParams.hh"
#include "physics/material/MaterialParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with processes and helper classes.
 */
PhysicsParams::PhysicsParams(Input inp) : processes_(std::move(inp.processes))
{
    CELER_EXPECT(!processes_.empty());
    CELER_EXPECT(std::all_of(processes_.begin(),
                             processes_.end(),
                             [](const SPConstProcess& p) { return bool(p); }));
    CELER_EXPECT(inp.particles);
    CELER_EXPECT(inp.materials);

    // Resize: processes for each particle
    processes_by_particle_.resize(inp.particles->size());

    // Construct models, assigning each model ID
    ModelIdGenerator next_model_id;
    for (auto process_idx : range<ProcessId::value_type>(processes_.size()))
    {
        auto new_models = processes_[process_idx]->build_models(next_model_id);
        CELER_ASSERT(!new_models.empty());
        for (const SPConstModel& model : new_models)
        {
            ModelId model_id = next_model_id();
            CELER_ASSERT(model->model_id() == model_id);

            // TODO: Add processes to particle-per-process
        }

        // TODO: Loop over materials and applicable particles, and call
        // step_limits with each particle type and material ID to get physics
        // tables

        // Move models to the end of the vector
        celeritas::move_extend(std::move(new_models), &models_);
    }

    CELER_NOT_IMPLEMENTED("constructing physics params");
}

//---------------------------------------------------------------------------//
} // namespace celeritas
