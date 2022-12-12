//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/RunAction.cc
//---------------------------------------------------------------------------//
#include "RunAction.hh"

#include <CLHEP/Random/Random.h>
#include <G4AutoLock.hh>
#include <G4Run.hh>
#include <G4TransportationManager.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/CutoffParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/ProcessBuilder.hh"
#include "celeritas/random/RngParams.hh"
#include "celeritas/track/TrackInitParams.hh"

namespace
{
G4Mutex mutex = G4MUTEX_INITIALIZER;
}

namespace celeritas
{
G4ThreadLocal RunData::SPTransporter RunData::transport = nullptr;

//---------------------------------------------------------------------------//
/*!
 * Construct with Celeritas setup options and shared data.
 */
RunAction::RunAction(SPCOptions options, SPData data)
    : options_(options), data_(data)
{
    CELER_EXPECT(options_);
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize Celeritas.
 */
void RunAction::BeginOfRunAction(const G4Run* run)
{
    CELER_EXPECT(run);
    CELER_LOG_LOCAL(debug) << "RunAction::BeginOfRunAction for run "
                           << run->GetRunID()
                           << (this->IsMaster() ? " (master)" : "");

    if (!data_->params)
    {
        // Maybe the first thread to run: build and store core params
        this->build_core_params();
    }
    CELER_ASSERT(data_->params);

    // Construct thread-local transporter
    data_->transport
        = std::make_shared<RunData::Transporter>(options_, data_->params);
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void RunAction::EndOfRunAction(const G4Run*)
{
    CELER_LOG_LOCAL(debug) << "RunAction::EndOfRunAction";
}

//---------------------------------------------------------------------------//
/*!
 * Finalize Celeritas.
 */
void RunAction::build_core_params()
{
    G4AutoLock lock(&mutex);
    if (data_->params)
    {
        // Some other thread constructed params between the thread-unsafe check
        // and this thread-safe check
        return;
    }

    celeritas::GeantImporter load_geant_data(
        G4TransportationManager::GetTransportationManager()
            ->GetNavigatorForTracking()
            ->GetWorldVolume());

    auto imported = load_geant_data();
    CELER_LOG_LOCAL(info) << "loaded data: "
                          << (imported ? "success" : "failure");

    CoreParams::Input params;

    // Create action manager
    {
        params.action_reg = std::make_shared<ActionRegistry>();
    }

    // Load geometry
    {
        params.geometry = std::make_shared<GeoParams>(options_->geometry_file);
        if (!params.geometry->supports_safety())
        {
            CELER_LOG(warning)
                << "Geometry contains surfaces that are "
                   "incompatible with the current ORANGE simple "
                   "safety algorithm: multiple scattering may "
                   "result in arbitrarily small steps";
        }
    }

    // Load materials
    {
        params.material = MaterialParams::from_import(imported);
    }

    // Create geometry/material coupling
    {
        params.geomaterial = GeoMaterialParams::from_import(
            imported, params.geometry, params.material);
    }

    // Construct particle params
    {
        params.particle = ParticleParams::from_import(imported);
    }

    // Construct cutoffs
    {
        params.cutoff = CutoffParams::from_import(
            imported, params.particle, params.material);
    }

    // Load physics: create individual processes with make_shared
    {
        PhysicsParams::Input input;
        input.particles       = params.particle;
        input.materials       = params.material;
        input.action_registry = params.action_reg.get();

        input.options.linear_loss_limit = imported.em_params.linear_loss_limit;
        input.options.secondary_stack_factor = options_->secondary_stack_factor;

        {
            ProcessBuilder::Options opts;
            ProcessBuilder          build_process(
                imported, opts, params.particle, params.material);

            std::set<ImportProcessClass> all_process_classes;
            for (const auto& p : imported.processes)
            {
                all_process_classes.insert(p.process_class);
            }
            for (auto p : all_process_classes)
            {
                input.processes.push_back(build_process(p));
            }
        }

        params.physics = std::make_shared<PhysicsParams>(std::move(input));
    }

    // TODO: different along-step kernels
    {
        // Create along-step action
        auto along_step = AlongStepGeneralLinearAction::from_params(
            params.action_reg->next_id(),
            *params.material,
            *params.particle,
            *params.physics,
            imported.em_params.energy_loss_fluct);
        params.action_reg->insert(along_step);
    }

    // Construct RNG params
    {
        params.rng
            = std::make_shared<RngParams>(CLHEP::HepRandom::getTheSeed());
    }

    // Construct track initialization params
    {
        TrackInitParams::Input input;
        input.capacity   = options_->initializer_capacity;
        input.max_events = options_->max_num_events;
        params.init      = std::make_shared<TrackInitParams>(input);
    }

    // Create params
    CELER_ASSERT(params);
    data_->params = std::make_shared<CoreParams>(std::move(params));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
