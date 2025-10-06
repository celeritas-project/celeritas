//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/UserActionIntegration.test.cc
//---------------------------------------------------------------------------//
#include "accel/UserActionIntegration.hh"

#include <memory>
#include <G4RunManager.hh>

#include "geocel/ScopedGeantExceptionHandler.hh"
#include "accel/SetupOptions.hh"
#include "accel/detail/IntegrationSingleton.hh"

#include "IntegrationTestBase.hh"
#include "celeritas_test.hh"

using UAI = celeritas::UserActionIntegration;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class UAITrackingAction final : public G4UserTrackingAction
{
    void PreUserTrackingAction(G4Track const* track) final
    {
        UAI::Instance().PreUserTrackingAction(const_cast<G4Track*>(track));
    }
};

//---------------------------------------------------------------------------//
class UAITestBase : virtual public IntegrationTestBase
{
    using Base = IntegrationTestBase;

  protected:
    void BeginOfRunAction(G4Run const* run) override
    {
        UAI::Instance().BeginOfRunAction(run);
    }
    void EndOfRunAction(G4Run const* run) override
    {
        UAI::Instance().EndOfRunAction(run);
    }
    void BeginOfEventAction(G4Event const* event) override
    {
        UAI::Instance().BeginOfEventAction(event);
    }
    void EndOfEventAction(G4Event const* event) override
    {
        UAI::Instance().EndOfEventAction(event);

        auto const& local_transport
            = detail::IntegrationSingleton::local_transporter();
        EXPECT_EQ(0, local_transport.GetBufferSize());
    }
    UPTrackAction make_tracking_action() override
    {
        return std::make_unique<UAITrackingAction>();
    }
};

//---------------------------------------------------------------------------//
class LarSphere : public LarSphereIntegrationMixin, public UAITestBase
{
};

TEST_F(LarSphere, run)
{
    auto& rm = this->run_manager();
    UAI::Instance().SetOptions(this->make_setup_options());

    cout << "initializing" << endl;
    rm.Initialize();
    cout << "beam on" << endl;

    rm.BeamOn(3);
    cout << "initial run done" << endl;
    rm.BeamOn(1);
    cout << "second run done" << endl;
}

//---------------------------------------------------------------------------//
class TestEm3 : public TestEm3IntegrationMixin, public UAITestBase
{
};

TEST_F(TestEm3, run)
{
    // Test loading instance before run manager
    auto& uai = UAI::Instance();

    {
        // Options can't be set before run manager is initialized
        ScopedGeantExceptionHandler convert_to_throw;
        EXPECT_THROW(uai.SetOptions(SetupOptions{}), RuntimeError);
    }

    auto& rm = this->run_manager();
    // Set options for real
    uai.SetOptions(this->make_setup_options());

    rm.Initialize();
    rm.BeamOn(2);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
