//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Cerenkov.test.cc
//---------------------------------------------------------------------------//
#include "OpticalTestBase.hh"

#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleTrackView.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Construct by populating particle params.
 */
OpticalTestBase::OpticalTestBase()
{
    units::MevMass e_mass(0.5109989461);

    ParticleParams::Input inp;
    inp.push_back({"electron",
                   pdg::electron(),
                   e_mass,
                   units::ElementaryCharge{-1},
                   constants::stable_decay_constant});
    inp.push_back({"positron",
                   pdg::positron(),
                   e_mass,
                   units::ElementaryCharge{1},
                   constants::stable_decay_constant});
    particle_params_ = std::make_shared<ParticleParams>(std::move(inp));

    resize(&state_val_, particle_params_->host_ref(), 1);
    state_ref_ = state_val_;
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor.
 */
OpticalTestBase::~OpticalTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Initialize particle state with given energy.
 */
ParticleTrackView
OpticalTestBase::make_particle_track_view(units::MevEnergy energy,
                                          PDGNumber pdg)
{
    ParticleTrackView::Initializer_t init_track;
    init_track.particle_id = particle_params_->find(pdg);
    init_track.energy = energy;

    ParticleTrackView particle_view(
        particle_params_->host_ref(), state_ref_, TrackSlotId(0));
    particle_view = init_track;
    return particle_view;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
