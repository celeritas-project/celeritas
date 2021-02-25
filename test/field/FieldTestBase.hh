//--------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldTestBase.hh
//---------------------------------------------------------------------------//
#include "geometry/GeoInterface.hh"
#include "geometry/GeoParams.hh"
#include "geometry/GeoTrackView.hh"

#include "base/PieStateStore.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/ParticleInterface.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Units.hh"

#include "field/FieldParamsPointers.hh"
#include "FieldTestParams.hh"

#include <VecGeom/navigation/NavigationState.h>

#include "celeritas_test.hh"

namespace celeritas_test
{
using namespace celeritas;
using namespace celeritas::units;

class FieldTestBase : public Test
{
  public:
    using NavState = vecgeom::cxx::NavigationState;

    void SetUp()
    {
        // Set up shared geometry data
        std::string gdml_file
            = celeritas::Test::test_data_path("field", "fieldTest.gdml");
        geo_params = std::make_shared<GeoParams>(gdml_file.c_str());

        int max_depth = geo_params->max_depth();
        state.reset(NavState::MakeInstance(max_depth));
        next_state.reset(NavState::MakeInstance(max_depth));

        geo_state_view.size       = 1;
        geo_state_view.vgmaxdepth = max_depth;
        geo_state_view.pos        = &this->pos;
        geo_state_view.dir        = &this->dir;
        geo_state_view.next_step  = &this->next_step;
        geo_state_view.vgstate    = this->state.get();
        geo_state_view.vgnext     = this->next_state.get();

        // Construct geometry parameter views
        geo_params_view = geo_params->host_pointers();
        CELER_ASSERT(geo_params_view.world_volume);

        // Set up shared particle data
        constexpr auto stable = ParticleDef::stable_decay_constant();

        ParticleParams::Input defs;
        defs.push_back({"electron",
                        pdg::electron(),
                        MevMass{0.5109989461},
                        ElementaryCharge{-1},
                        stable});
        defs.push_back({"positron",
                        pdg::positron(),
                        MevMass{0.5109989461},
                        ElementaryCharge{1},
                        stable});

        particle_params = std::make_shared<ParticleParams>(std::move(defs));

        // Construct particle parameter views
        resize(&state_value, particle_params->host_pointers(), 1);
        state_ref = state_value;

        // Set values of FieldParamsPointers;
        field_params_view.delta_chord           = 0.25;
        field_params_view.delta_intersection    = 1.0e-4;
        field_params_view.epsilon_step          = 1.0e-5;
        field_params_view.minimun_step          = 1.0e-5;
        field_params_view.safety                = 0.9;
        field_params_view.pgrow                 = -0.20;
        field_params_view.pshrink               = -0.25;
        field_params_view.errcon                = 1.0e-4;
        field_params_view.max_stepping_increase = 5;
        field_params_view.max_stepping_decrease = 0.1;
        field_params_view.max_nsteps            = 100;

        // Input parameters of an electron in a uniform magnetic field
        // XXX: make this as an input argument with v
        test_params.nstates     = 32 * 32;
        test_params.nsteps      = 12;
        test_params.revolutions = 10;
        test_params.field_value = 0.001;
        test_params.radius      = 38.085386;
        test_params.energy      = 10.9181415106;
        test_params.momentum    = 11.417711;
        test_params.epsilon     = 1.0e-5;
    }

  protected:
    // State data
    Real3                     pos;
    Real3                     dir;
    real_type                 next_step;
    std::unique_ptr<NavState> state;
    std::unique_ptr<NavState> next_state;

    std::shared_ptr<GeoParams>      geo_params;
    std::shared_ptr<ParticleParams> particle_params;

    ParticleStateData<Ownership::value, MemSpace::host>     state_value;
    ParticleStateData<Ownership::reference, MemSpace::host> state_ref;

    // Views
    GeoParamsPointers   geo_params_view;
    GeoStatePointers    geo_state_view;
    FieldParamsPointers field_params_view;

    // Test parameters
    FieldTestParams test_params;
};

} // namespace celeritas_test
