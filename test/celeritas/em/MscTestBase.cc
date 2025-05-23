//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/MscTestBase.cc
//---------------------------------------------------------------------------//
#include "MscTestBase.hh"

#include "celeritas/geo/GeoParams.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/track/SimParams.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Initialize states on construction.
 */
MscTestBase::MscTestBase()
{
    size_type state_size = 1;
    physics_state_ = StateStore<PhysicsStateData>(this->physics()->host_ref(),
                                                  state_size);
    particle_state_ = StateStore<ParticleStateData>(
        this->particle()->host_ref(), state_size);
    geo_state_
        = StateStore<GeoStateData>(this->geometry()->host_ref(), state_size);
    sim_state_ = StateStore<SimStateData>(this->sim()->host_ref(), state_size);
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor.
 */
MscTestBase::~MscTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Access particle state data.
 */
ParticleTrackView
MscTestBase::make_par_view(PDGNumber pdg, MevEnergy energy) const
{
    CELER_EXPECT(pdg);
    CELER_EXPECT(energy > zero_quantity());
    auto pid = this->particle()->find(pdg);
    CELER_ASSERT(pid);

    ParticleTrackView par{
        this->particle()->host_ref(), particle_state_.ref(), TrackSlotId{0}};
    ParticleTrackView::Initializer_t init;
    init.particle_id = pid;
    init.energy = energy;
    par = init;
    return par;
}

//---------------------------------------------------------------------------//
/*!
 * Access particle state data.
 */
PhysicsTrackView
MscTestBase::make_phys_view(ParticleTrackView const& par,
                            std::string const& matname,
                            HostCRef<PhysicsParamsData> const& host_ref) const
{
    auto mid = this->material()->find_material(matname);
    CELER_ASSERT(mid);

    // Initialize physics
    PhysicsTrackView phys_view(
        host_ref, physics_state_.ref(), par, mid, TrackSlotId{0});
    phys_view = PhysicsTrackInitializer{};

    // Calculate and store the energy loss (dedx) range limit
    auto grid_id = phys_view.range_grid();
    auto calc_range = phys_view.make_calculator<RangeCalculator>(grid_id);
    real_type range = calc_range(par.energy());
    phys_view.dedx_range(range);

    return phys_view;
}

//---------------------------------------------------------------------------//
/*!
 * Access geometry state data.
 */
GeoTrackView MscTestBase::make_geo_view(real_type r) const
{
    GeoTrackView geo_view(
        this->geometry()->host_ref(), geo_state_.ref(), TrackSlotId{0});
    geo_view = {{r, r, r}, Real3{0, 0, 1}};
    return geo_view;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
