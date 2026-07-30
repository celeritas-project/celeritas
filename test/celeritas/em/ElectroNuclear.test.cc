//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/ElectroNuclear.test.cc
//---------------------------------------------------------------------------//
#include <memory>

#include "celeritas_test_config.h"

#include "corecel/cont/Range.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/interactor/ElectroNuclearInteractor.hh"
#include "celeritas/em/model/ElectroNuclearModel.hh"
#include "celeritas/em/xs/ElectroNuclearMicroXsCalculator.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/InteractorHostTestBase.hh"
#include "celeritas/phys/MacroXsCalculator.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class ElectroNuclearTest : public InteractorHostTestBase
{
  protected:
    using MevEnergy = units::MevEnergy;

    void SetUp() override
    {
        using namespace units;

        // Set up the default particle: 1000 MeV electron along +z direction
        auto const& particles = *this->particle_params();
        this->set_inc_particle(pdg::electron(), MevEnergy{1000});
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

        model_ = std::make_shared<ElectroNuclearModel>(
            ActionId{0}, particles, *this->material_params());

        this->set_material("PbWO4");
    }

  protected:
    std::shared_ptr<ElectroNuclearModel> model_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(ElectroNuclearTest, micro_xs)
{
    // Calculate the electro-nuclear microscopic (element) cross section
    using XsCalculator = ElectroNuclearMicroXsCalculator;

    // Set the target element: Pb
    ElementId el_id{2};

    // Check the size of the element cross section data
    HostCRef<ElectroNuclearData> shared = model_->host_ref();

    // Calculate the electro-nuclear cross section using parameterizated data

    NonuniformGridRecord grid = shared.micro_xs[el_id];
    EXPECT_EQ(grid.grid.size(), 300);

    // Expected microscopic cross section (units::BarnXs) in [100:1e+8] (MeV)
    std::vector<std::pair<real_type, real_type>> const energy_xs
        = {{200, 0.0076866011595330607},
           {500, 0.010594901781001968},
           {1e+3, 0.012850271316595725},
           {5e+3, 0.018062295361634773},
           {5e+4, 0.025265243621752368},
           {1e+6, 0.035225631593952887},
           {1e+7, 0.044179324718181687},
           {1e+8, 0.05492003274983899}};

    for (auto i : range(energy_xs.size()))
    {
        XsCalculator calc_micro_xs(shared, MevEnergy{energy_xs[i].first});
        EXPECT_SOFT_EQ(calc_micro_xs(el_id).value(), energy_xs[i].second);
    }
}

TEST_F(ElectroNuclearTest, macro_xs)
{
    // Calculate the electro-nuclear macroscopic cross section
    auto material = this->material_track().material_record();
    auto calc_xs = MacroXsCalculator<ElectroNuclearMicroXsCalculator>(
        model_->host_ref(), material);

    // Expected macroscopic cross section (\f$ cm^{-1} \f$)} in [100:1e+8](MeV)
    std::vector<std::pair<real_type, real_type>> const energy_xs
        = {{200, 0.03099484247367704},
           {500, 0.042848174635210838},
           {1e+3, 0.05204849281161035},
           {5e+3, 0.073325144774783371},
           {5e+4, 0.10275850385625593},
           {1e+6, 0.14353937451738991},
           {1e+7, 0.1802859228767722},
           {1e+8, 0.22446330065829731}};

    for (auto i : range(energy_xs.size()))
    {
        EXPECT_SOFT_EQ(native_value_to<units::InvCmXs>(
                           calc_xs(MevEnergy{energy_xs[i].first}))
                           .value(),
                       energy_xs[i].second);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
