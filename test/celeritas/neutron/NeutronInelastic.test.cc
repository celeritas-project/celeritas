//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/NeutronBertini.test.cc
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
        NeutronXsReader read_el_data(data_path.c_str(), NeutronXsType::inel);

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
    NeutronInelasticRef shared = model_->host_ref();
    GenericGridData grid = shared.micro_xs[el_id];
    EXPECT_EQ(grid.grid.size(), 299);

    // Microscopic cross section (units::BarnXs) in [1e-04:1e+4] (MeV)
    std::vector<real_type> const expected_micro_xs = {2.499560005640001e-06,
                                                      2.499560005640001e-06,
                                                      2.499560005640001e-06,
                                                      2.499560005640001e-06,
                                                      0.21345480858091287,
                                                      1.3683015986762637,
                                                      0.81011125346487955,
                                                      0.84791989750495067};

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
    std::vector<real_type> const expected_macro_xs = {1.0538237187534606e-06,
                                                      4.4336422193095875e-07,
                                                      2.510056299100037e-07,
                                                      1.925905724096256e-07,
                                                      0.014808449342475376,
                                                      0.094926010256081828,
                                                      0.05684884713429017,
                                                      0.059657133034676249};

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
        0.061235054332723221);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
