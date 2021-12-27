//--------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldTestBase.hh
//---------------------------------------------------------------------------//
#include "base/CollectionStateStore.hh"
#include "physics/base/Units.hh"

#include "geometry/GeoData.hh"
#include "geometry/GeoParams.hh"
#include "geometry/GeoTrackView.hh"

#include "physics/base/ParticleParams.hh"
#include "physics/base/ParticleData.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Units.hh"

#include "field/FieldParamsData.hh"
#include "field/FieldTestParams.hh"

#include "celeritas_test.hh"

namespace celeritas_test
{
using namespace celeritas;
using namespace celeritas::units;

using celeritas::ParticleId;
using celeritas::ParticleParams;
using celeritas::ParticleTrackView;

using celeritas::real_type;
using celeritas::ThreadId;
using celeritas::units::MevEnergy;

class FieldTestBase : public celeritas::Test
{
  public:
    using GeoStateStore = CollectionStateStore<GeoStateData, MemSpace::host>;

    void SetUp()
    {
        // geo track view
        std::string test_file
            = celeritas::Test::test_data_path("field", "field-test.gdml");
        geo_params = std::make_shared<celeritas::GeoParams>(test_file.c_str());
        geo_state  = GeoStateStore(*this->geo_params, 1);

        // particle track view
        namespace pdg = celeritas::pdg;
        using namespace celeritas::units;

        constexpr auto stable = ParticleDef::stable_decay_constant();

        // Create particle defs, initialize on device
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

        // Construct views
        resize(&state_value, particle_params->host_ref(), 1);
        state_ref = state_value;

        // Set values of FieldParamsData;
        field_params.delta_intersection = 1.0e-4 * units::millimeter;

        // Input parameters of an electron in a uniform magnetic field
        test.nstates     = 128;
        test.nsteps      = 100;
        test.revolutions = 10;
        test.field_value = 1.0 * units::tesla;
        test.radius      = 3.8085386036;
        test.delta_z     = 6.7003310629;
        test.energy      = 10.9181415106;
        test.momentum_y  = 11.4177114158018;
        test.momentum_z  = 0.0;
        test.epsilon     = 1.0e-5;
    }

  protected:
    GeoStateStore                               geo_state;
    std::shared_ptr<const celeritas::GeoParams> geo_params;

    std::shared_ptr<ParticleParams>                         particle_params;
    ParticleStateData<Ownership::value, MemSpace::host>     state_value;
    ParticleStateData<Ownership::reference, MemSpace::host> state_ref;

    FieldParamsData field_params;

    // Test parameters
    FieldTestParams test;
};

} // namespace celeritas_test
