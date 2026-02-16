//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SurfacePhysicsIntegrationTestBase.cc
//---------------------------------------------------------------------------//
#include "SurfacePhysicsIntegrationTestBase.hh"

#include "geocel/UnitUtils.hh"
#include "celeritas/optical/TrackInitializer.hh"

using ::celeritas::test::from_cm;

namespace celeritas
{
namespace optical
{
namespace test
{
//---------------------------------------------------------------------------//
GeantPhysicsOptions
SurfacePhysicsIntegrationTestBase::build_geant_options() const
{
    auto result = GeantTestBase::build_geant_options();
    result.optical = {};
    CELER_ENSURE(result.optical);
    return result;
}

//---------------------------------------------------------------------------//
GeantImportDataSelection
SurfacePhysicsIntegrationTestBase::build_import_data_selection() const
{
    auto result = GeantTestBase::build_import_data_selection();
    result.processes |= GeantImportDataSelection::optical;
    return result;
}

//---------------------------------------------------------------------------//
auto SurfacePhysicsIntegrationTestBase::select_optical_models() const
    -> std::vector<IMC>
{
    return {IMC::absorption};
}

//---------------------------------------------------------------------------//
auto SurfacePhysicsIntegrationTestBase::build_optical_surface_physics()
    -> SPConstOpticalSurfacePhysics
{
    inp::SurfacePhysics input;

    this->setup_surface_models(input);

    return std::make_shared<SurfacePhysicsParams>(
        this->optical_action_reg().get(), input);
}

//---------------------------------------------------------------------------//
void SurfacePhysicsIntegrationTestBase::initialize_run()
{
    generate_ = DirectGeneratorAction::make_and_insert(*this->optical_params());

    Transporter::Input inp;
    inp.params = this->optical_params();
    transport_ = std::make_shared<Transporter>(std::move(inp));

    size_type num_tracks = 128;
    auto state = std::make_shared<CoreState<MemSpace::host>>(
        *this->optical_params(), StreamId{0}, num_tracks);
    state->aux() = std::make_shared<AuxStateVec>(
        *this->core()->aux_reg(), MemSpace::host, StreamId{0}, num_tracks);
    state_ = state;
}

//---------------------------------------------------------------------------//
void SurfacePhysicsIntegrationTestBase::run_step(RealTurn angle)
{
    real_type sin_theta;
    real_type cos_theta;
    sincos(angle, &sin_theta, &cos_theta);

    std::vector<TrackInitializer> inits(
        100,
        TrackInitializer{units::MevEnergy{3e-6},
                         from_cm(Real3{0, 49, 0}),
                         Real3{sin_theta, cos_theta, 0},  // direction
                         Real3{0, 0, 1},  // polarization
                         0,  // time
                         {},  // primary
                         ImplVolumeId{0}});

    generate_->insert(*state_, make_span(inits));

    (*transport_)(*state_);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
