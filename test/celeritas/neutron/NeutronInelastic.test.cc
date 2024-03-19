//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/NeutronInelastic.test.cc
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
#include "celeritas/neutron/interactor/NeutronInelasticInteractor.hh"
#include "celeritas/neutron/model/NeutronInelasticModel.hh"
#include "celeritas/neutron/xs/NeutronInelasticMicroXsCalculator.hh"
#include "celeritas/phys/MacroXsCalculator.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class NeutronInelasticTest : public NeutronTestBase
{
  protected:
    using MevEnergy = units::MevEnergy;
    using SPConstNInelasticModel = std::shared_ptr<NeutronInelasticModel const>;

    void SetUp() override
    {
        using namespace units;

        // Load neutron elastic cross section data
        std::string data_path = this->test_data_path("celeritas", "");
        NeutronXsReader read_el_data(NeutronXsType::inel, data_path.c_str());

        // Set up the default particle: 100 MeV neutron along +z direction
        auto const& particles = *this->particle_params();
        this->set_inc_particle(pdg::neutron(), MevEnergy{100});
        this->set_inc_direction({0, 0, 1});

        // Set up the default material
        this->set_material("HeCu");
        model_ = std::make_shared<NeutronInelasticModel>(
            ActionId{0}, particles, *this->material_params(), read_el_data);
    }

  protected:
    SPConstNInelasticModel model_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(NeutronInelasticTest, micro_xs)
{
    // Calculate the elastic neutron-nucleus microscopic cross section
    using XsCalculator = NeutronInelasticMicroXsCalculator;

    // Set the target element: Cu
    ElementId el_id{1};

    // Check the size of the element cross section data (G4PARTICLEXS4.0)
    // The neutron/inelZ data are pruned by a factor of 5 for this test
    NeutronInelasticRef shared = model_->host_ref();
    GenericGridData grid = shared.micro_xs[el_id];
    EXPECT_EQ(grid.grid.size(), 61);

    // Microscopic cross section (units::BarnXs) in [1e-04:1e+4] (MeV)
    std::vector<real_type> const expected_micro_xs = {2.499560005640001e-06,
                                                      2.499560005640001e-06,
                                                      2.499560005640001e-06,
                                                      2.499560005640001e-06,
                                                      0.2170446680979802,
                                                      1.3677671823188946,
                                                      0.81016638725225387,
                                                      0.84789596907525477};

    real_type energy = 1e-4;
    real_type const factor = 1e+1;
    for (auto i : range(expected_micro_xs.size()))
    {
        XsCalculator calc_micro_xs(shared, MevEnergy{energy});
        EXPECT_SOFT_EQ(calc_micro_xs(el_id).value(), expected_micro_xs[i]);
        energy *= factor;
    }

    // Check the elastic cross section at the upper bound (20 GeV)
    XsCalculator calc_upper_xs(shared, MevEnergy{2e+4});
    EXPECT_SOFT_EQ(calc_upper_xs(el_id).value(), 0.80300000000000027);
}

TEST_F(NeutronInelasticTest, macro_xs)
{
    // Calculate the inelastic neutron-nucleus macroscopic cross section
    auto material = this->material_track().make_material_view();
    auto calc_xs = MacroXsCalculator<NeutronInelasticMicroXsCalculator>(
        model_->host_ref(), material);

    // Macroscopic cross section (\f$ cm^{-1} \f$) in [1e-04:1e+4] (MeV)
    std::vector<real_type> const expected_macro_xs = {1.0577605656430734e-06,
                                                      4.4447010621996484e-07,
                                                      2.5134945234021254e-07,
                                                      1.9270371228950039e-07,
                                                      0.015057496086707027,
                                                      0.094888935102106969,
                                                      0.056850427191922973,
                                                      0.059657345679963072};

    real_type energy = 1e-4;
    real_type const factor = 1e+1;
    for (auto i : range(expected_macro_xs.size()))
    {
        EXPECT_SOFT_EQ(
            native_value_to<units::InvCmXs>(calc_xs(MevEnergy{energy})).value(),
            expected_macro_xs[i]);
        energy *= factor;
    }

    // Check the neutron inelastic interaction cross section at the upper bound
    // (20 GeV)
    EXPECT_SOFT_EQ(
        native_value_to<units::InvCmXs>(calc_xs(MevEnergy{2000})).value(),
        0.061219850473480573);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
