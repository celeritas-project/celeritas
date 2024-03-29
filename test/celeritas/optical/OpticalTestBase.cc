//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Cerenkov.test.cc
//---------------------------------------------------------------------------//
#include "OpticalTestBase.hh"

#include "geocel/UnitUtils.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/track/SimParams.hh"
#include "celeritas/track/SimTrackView.hh"

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

    particle_state_
        = StateStore<ParticleStateData>(particle_params_->host_ref(), 1);
    sim_state_ = StateStore<SimStateData>(1);
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor.
 */
OpticalTestBase::~OpticalTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Resize distributions.
 */
void OpticalTestBase::resize_distributions(int count)
{
    CELER_EXPECT(count > 0);
    distributions_ = StateStore<DistributionStackData>(count);
    distribution_allocator_
        = std::make_shared<DistributionAllocator>(distributions_.ref());
}

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
        particle_params_->host_ref(), particle_state_.ref(), TrackSlotId(0));
    particle_view = init_track;
    return particle_view;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize sim track state with step length with a step lenght in [cm].
 */
SimTrackView OpticalTestBase::make_sim_track_view(real_type step_len_cm)
{
    SimTrackView::Initializer_t init_track;
    init_track.event_id = EventId{0};
    init_track.parent_id = TrackId{0};
    init_track.status = TrackStatus::alive;

    SimTrackView sim_view(
        sim_params_->host_ref(), sim_state_.ref(), TrackSlotId(0));
    sim_view = init_track;
    sim_view.step_length(from_cm(step_len_cm));
    return sim_view;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
