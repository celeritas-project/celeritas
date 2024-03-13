//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/NeutronCapture.test.cc
//---------------------------------------------------------------------------//
#include <memory>

#include "corecel/cont/Range.hh"
#include "corecel/data/Ref.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/grid/GenericGridData.hh"
#include "celeritas/io/NeutronXsReader.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/neutron/NeutronTestBase.hh"
#include "celeritas/neutron/model/NeutronCaptureModel.hh"
#include "celeritas/neutron/xs/NeutronCaptureMicroXsCalculator.hh"
#include "celeritas/phys/MacroXsCalculator.hh"

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

        // Load neutron capture cross section data
        std::string data_path = this->test_data_path("celeritas", "");
        NeutronXsReader read_el_data(data_path.c_str(), NeutronXsType::cap);

        // Set up the default particle: 1 MeV neutron along +z direction
        auto const& particles = *this->particle_params();
        this->set_inc_particle(pdg::neutron(), MevEnergy{1});
        this->set_inc_direction({0, 0, 1});

        // Set up the default material
        this->set_material("HeCu");
        model_ = std::make_shared<NeutronCaptureModel>(
            ActionId{0}, particles, *this->material_params(), read_el_data);
    }

  protected:
    SPConstNCaptureModel model_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(NeutronCaptureTest, micro_xs)
{
    // Calculate the neutron capture microscopic cross section
    using XsCalculator = NeutronCaptureMicroXsCalculator;

    // Set the target element: Cu
    ElementId el_id{1};

    // Check the size of the element cross section data (G4PARTICLEXS4.0)
    NeutronCaptureRef shared = model_->host_ref();
    GenericGridData grid = shared.micro_xs[el_id];
    EXPECT_EQ(grid.grid.size(), 1201);

    // Microscopic cross section (units::BarnXs) in [1e-5:20] (MeV)
    std::vector<real_type> const expected_micro_xs = {0.18204728405100004,
                                                      0.042623832551870709,
                                                      0.30824208140316706,
                                                      0.34512476672509756,
                                                      0.026301149066701626,
                                                      0.013598516329341233,
                                                      0.00090382863741660178};

    real_type energy = 1e-5;
    real_type const factor = 1e+1;
    for (auto i : range(expected_micro_xs.size()))
    {
        XsCalculator calc_micro_xs(shared, MevEnergy{energy});
        EXPECT_SOFT_EQ(calc_micro_xs(el_id).value(), expected_micro_xs[i]);
        energy *= factor;
    }

    // Check the capture section at the upper bound (20 MeV)
    XsCalculator calc_upper_xs(shared, MevEnergy{20});
    EXPECT_SOFT_EQ(calc_upper_xs(el_id).value(), 0.00059643802900000023);
}

TEST_F(NeutronCaptureTest, macro_xs)
{
    // Calculate the neutron capture macroscopic cross section
    auto material = this->material_track().make_material_view();
    auto calc_xs = MacroXsCalculator<NeutronCaptureMicroXsCalculator>(
        model_->host_ref(), material);

    // Microscopic cross section (\f$ cm^{-1} \f$) in [1e-5:20] (MeV)
    std::vector<real_type> const expected_macro_xs = {0.012629541537026693,
                                                      0.002957031007021316,
                                                      0.021384313371274329,
                                                      0.023943051935853676,
                                                      0.0018246438357136993,
                                                      0.00094339790818402766,
                                                      6.2703167786389206e-05};

    real_type energy = 1e-5;
    real_type const factor = 1e+1;
    for (auto i : range(expected_macro_xs.size()))
    {
        EXPECT_SOFT_EQ(
            native_value_to<units::InvCmXs>(calc_xs(MevEnergy{energy})).value(),
            expected_macro_xs[i]);
        energy *= factor;
    }

    // Check the neutron capture cross section at the upper bound (20 MeV)
    EXPECT_SOFT_EQ(
        native_value_to<units::InvCmXs>(calc_xs(MevEnergy{20})).value(),
        4.1377925277976458e-05);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
