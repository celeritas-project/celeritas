//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/GammaNuclear.test.cc
//---------------------------------------------------------------------------//
#include <memory>

#include "celeritas_test_config.h"

#include "corecel/cont/Range.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/interactor/GammaNuclearInteractor.hh"
#include "celeritas/em/model/GammaNuclearModel.hh"
#include "celeritas/em/xs/GammaNuclearMicroXsCalculator.hh"
#include "celeritas/io/GammaNuclearXsReader.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/InteractorHostTestBase.hh"
#include "celeritas/phys/MacroXsCalculator.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class GammaNuclearTest : public InteractorHostTestBase
{
  protected:
    using MevEnergy = units::MevEnergy;

    void SetUp() override
    {
        using namespace units;

        // Load gamma-nuclear cross section data
        std::string path = celeritas_source_dir;
        path += "/test/celeritas/data/gamma-nucl/";
        GammaNuclearXsReader read_data(path.c_str());

        // Set up the default particle: 100 MeV gamma along +z direction
        auto const& particles = *this->particle_params();
        this->set_inc_particle(pdg::gamma(), MevEnergy{100});
        this->set_inc_direction({0, 0, 1});

        // Build the model with the default material
        MaterialParams::Input mi;
        mi.elements = {
            {AtomicNumber{8}, AmuMass{15.999}, {}, Label{"O"}},
            {AtomicNumber{74}, AmuMass{183.84}, {}, Label{"W"}},
            {AtomicNumber{82}, AmuMass{207.2}, {}, Label{"Pb"}},
        };

        mi.materials = {
            {native_value_from(MolCcDensity{8.28}),
             293.,
             MatterState::solid,
             {{ElementId{0}, 0.14}, {ElementId{1}, 0.4}, {ElementId{2}, 0.46}},
             Label{"PbWO4"}}};
        this->set_material_params(mi);

        model_ = std::make_shared<GammaNuclearModel>(
            ActionId{0}, particles, *this->material_params(), read_data);

        this->set_material("PbWO4");
    }

  protected:
    std::shared_ptr<GammaNuclearModel> model_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(GammaNuclearTest, micro_xs)
{
    // Calculate the gamma-nuclear microscopic (element) cross section
    using XsCalculator = GammaNuclearMicroXsCalculator;

    // Set the target element: Pb
    ElementId el_id{2};

    // Check the size of the element cross section data (G4PARTICLEXS4.1)
    HostCRef<GammaNuclearData> shared = model_->host_ref();
    NonuniformGridRecord grid = shared.micro_xs[el_id];
    EXPECT_EQ(grid.grid.size(), 260);

    // Microscopic cross section (units::BarnXs) in [0.5:100.5] (MeV)
    std::vector<real_type> const expected_micro_xs = {0,
                                                      0.067392400000000019,
                                                      0.016010000000000003,
                                                      0.015699200000000003,
                                                      0.015388000000000004,
                                                      0.013824900000000005};

    real_type energy = 0.5;
    real_type const factor = 2e+1;
    for (auto i : range(expected_micro_xs.size()))
    {
        XsCalculator calc_micro_xs(shared, MevEnergy{energy});
        EXPECT_SOFT_EQ(calc_micro_xs(el_id).value(), expected_micro_xs[i]);
        energy += factor;
    }

    // Check the gamma-nuclear element cross section at the upper bound
    XsCalculator calc_upper_xs(shared, MevEnergy{130});
    EXPECT_SOFT_EQ(calc_upper_xs(el_id).value(), 0.010895100000000003);
}

TEST_F(GammaNuclearTest, macro_xs)
{
    // Calculate the gamma nuclear macroscopic cross section
    auto material = this->material_track().material_record();
    auto calc_xs = MacroXsCalculator<GammaNuclearMicroXsCalculator>(
        model_->host_ref(), material);

    // Macroscopic cross section (\f$ cm^{-1} \f$) in [0.5:100.5] (MeV)
    std::vector<real_type> const expected_macro_xs = {0,
                                                      0.67518515551801506,
                                                      0.27924724815369489,
                                                      0.30744953728122743,
                                                      0.32743928018832685,
                                                      0.31806243520165606};

    real_type energy = 0.5;
    real_type const factor = 2e+1;
    for (auto i : range(expected_macro_xs.size()))
    {
        EXPECT_SOFT_EQ(
            native_value_to<units::InvCmXs>(calc_xs(MevEnergy{energy})).value(),
            expected_macro_xs[i]);
        energy += factor;
    }

    // Check the gamma-nuclear interaction cross section at the upper bound
    EXPECT_SOFT_EQ(
        native_value_to<units::InvCmXs>(calc_xs(MevEnergy{130})).value(),
        0.27716766602987458);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
