//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldPropagatorTestBase.hh
//---------------------------------------------------------------------------//

#include "celeritas/GlobalGeoTestBase.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "FieldTestParams.hh"
#include "celeritas_test.hh"

using namespace celeritas;
using namespace celeritas_test;
using celeritas::units::MevEnergy;

//---------------------------------------------------------------------------//
/*!
 * TestBase harness.
 *
 * The test geometry is 1-cm thick slabs normal to y, with 1-cm gaps in
 * between. We fire off electrons (TODO: also test positrons!) that end up
 * running circles around the z axis (along which the magnetic field points).
 */
class FieldPropagatorTestBase : public celeritas_test::GlobalGeoTestBase
{
  public:
    const char* geometry_basename() const override { return "field-test"; }

    SPConstMaterial build_material() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstGeoMaterial build_geomaterial() override
    {
        CELER_ASSERT_UNREACHABLE();
    }
    SPConstCutoff  build_cutoff() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstPhysics build_physics() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction  build_along_step() override { CELER_ASSERT_UNREACHABLE(); }

    SPConstParticle build_particle() override
    {
        using namespace celeritas::units;
        namespace pdg = celeritas::pdg;

        // Create particle defs
        constexpr auto        stable = ParticleRecord::stable_decay_constant();
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

        return std::make_shared<ParticleParams>(std::move(defs));
    }

    void SetUp() override
    {
        // Construct views
        resize(&state_value, this->particle()->host_ref(), 1);
        state_ref = state_value;

        // Set values of FieldDriverOptions;
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
    ::celeritas::HostVal<ParticleStateData> state_value;
    ::celeritas::HostRef<ParticleStateData> state_ref;

    FieldDriverOptions field_params;

    // Test parameters
    FieldTestParams test;
};
