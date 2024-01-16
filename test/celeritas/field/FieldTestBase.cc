//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldTestBase.cc
//---------------------------------------------------------------------------//
#include "FieldTestBase.hh"

#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Create a particle view with the given type and energy
ParticleTrackView
FieldTestBase::make_particle_view(PDGNumber pdg, MevEnergy energy)
{
    CELER_EXPECT(pdg && energy > zero_quantity());
    ParticleId pid = this->particle()->find(pdg);
    CELER_ASSERT(pid);
    ParticleTrackView view{
        this->particle()->host_ref(), par_state_.ref(), TrackSlotId{0}};
    view = {pid, energy};
    return view;
}

//---------------------------------------------------------------------------//
//! Access particle params (create if needed)
auto FieldTestBase::particle() -> SPConstParticle const&
{
    if (!particle_)
    {
        particle_ = this->build_particle();
        CELER_ASSERT(particle_);
        par_state_ = ParStateStore{particle_->host_ref(), 1};
    }
    return particle_;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
