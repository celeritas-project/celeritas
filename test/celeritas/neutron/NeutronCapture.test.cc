//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/NeutronCapture.test.cc
//---------------------------------------------------------------------------//
#include <memory>

#include "corecel/data/Ref.hh"
#include "corecel/grid/VectorUtils.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/neutron/model/NeutronCaptureModel.hh"
#include "celeritas/neutron/xs/NeutronCaptureMicroXsCalculator.hh"
#include "celeritas/phys/MacroXsCalculator.hh"

#include "NeutronTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class NeutronCaptureTest : public NeutronTestBase
{
  protected:
    using MevEnergy = units::MevEnergy;
    using SPConstNCaptureModel = std::shared_ptr<NeutronCaptureModel const>;

    void SetUp() override
    {
        using namespace units;

        // Load neutron capture cross section test data (filtered and reduced
        // by a factor of 5 from the original dataset)
        auto xs_reader = make_xs_reader(NeutronXsType::cap);

        // Set up the default particle: 100 MeV neutron along +z direction
        auto const& particles = *this->particle_params();
        this->set_inc_particle(pdg::neutron(), MevEnergy{100});
        this->set_inc_direction({0, 0, 1});

        // Set up the default material
        this->set_material("HeCu");
        model_ = std::make_shared<NeutronCaptureModel>(
            ActionId{0}, particles, *this->material_params(), xs_reader);
    }

  protected:
    SPConstNCaptureModel model_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(NeutronCaptureTest, micro_xs)
{
    // Calculate the neutron-nucleus capture microscopic cross section in the
    // valid range [1e-5:20] (MeV)
    using XsCalculator = NeutronCaptureMicroXsCalculator;

    // Set the target element: Cu
    ElementId el_id{1};

    // Check the size of the element cross section data (G4PARTICLEXS4.2)
    HostCRef<NeutronCaptureData> const& shared = model_->host_ref();
    EXPECT_EQ(shared.micro_xs[el_id].grid.size(), 72);

    // Microscopic cross section (units::BarnXs) in [1e-05:10] (MeV)
    std::vector<real_type> const expected_micro_xs = {0.182047284051,
                                                      0.042861326708155,
                                                      0.0087698603105641,
                                                      0.053880959700332,
                                                      0.024311284787054,
                                                      0.013574271991922,
                                                      0.00090554516078186,
                                                      0.000596438029};

    std::vector<real_type> micro_xs;
    for (auto energy : geomspace(1e-5, 10.0, 7))
    {
        XsCalculator calc_micro_xs(shared, MevEnergy(energy));
        micro_xs.push_back(calc_micro_xs(el_id).value());
    }

    // Check the capture cross section at the upper bound (20 MeV)
    XsCalculator calc_upper_xs(shared, MevEnergy(20));
    micro_xs.push_back(calc_upper_xs(el_id).value());

    EXPECT_VEC_SOFT_EQ(expected_micro_xs, micro_xs);
}

TEST_F(NeutronCaptureTest, macro_xs)
{
    // Calculate macroscopic cross sections of the neutron capture
    // (\f$ cm^{-1} \f$) in the valid range [1e-5:10] (MeV)
    auto material = this->material_track().material_record();
    auto calc_xs = MacroXsCalculator<NeutronCaptureMicroXsCalculator>(
        model_->host_ref(), material);

    std::vector<real_type> const expected_macro_xs = {0.012629541537022,
                                                      0.0029735071787424,
                                                      0.00060840959890254,
                                                      0.0037379948959393,
                                                      0.0016865968788258,
                                                      0.00094171595578094,
                                                      6.2822251701714e-05,
                                                      4.1377925277976e-05};

    std::vector<real_type> macro_xs;
    for (auto energy : geomspace(1e-5, 10.0, 7))
    {
        macro_xs.push_back(
            native_value_to<units::InvCmXs>(calc_xs(MevEnergy(energy))).value());
    }

    // Check the macroscopic cross section at the upper bound (20 MeV)
    macro_xs.push_back(
        native_value_to<units::InvCmXs>(calc_xs(MevEnergy(20))).value());

    EXPECT_VEC_SOFT_EQ(expected_macro_xs, macro_xs);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
