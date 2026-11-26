//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/woodcock/Woodcock.test.cc
//---------------------------------------------------------------------------//
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "corecel/cont/Range.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"
#include "celeritas/GeantTestBase.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/geo/CoreGeoParams.hh"
#include "celeritas/geo/CoreGeoTraits.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/inp/Tracking.hh"
#include "celeritas/inp/TrackingIO.json.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/woodcock/WoodcockParams.hh"
#include "celeritas/woodcock/WoodcockView.hh"
#include "celeritas/woodcock/detail/MajorantBuilder.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(WoodcockInputTest, json)
{
    inp::Tracking orig;
    EXPECT_FALSE(orig.woodcock);

    {
        auto j = nlohmann::json(orig);
        EXPECT_FALSE(j.get<inp::Tracking>().woodcock);
    }

    orig.woodcock.regions = {"foo", "bar"};
    orig.woodcock.min_energy = units::MevEnergy{0.5};
    orig.woodcock.max_flights = 3;
    EXPECT_TRUE(orig.woodcock);

    auto j = nlohmann::json(orig);
    EXPECT_JSON_EQ(
        R"json({"max_flights":3,"min_energy":[0.5,"MeV"],"regions":["foo","bar"]})json",
        j["woodcock"].dump());
    auto actual = j.get<inp::Tracking>();
    EXPECT_EQ(orig.woodcock.regions, actual.woodcock.regions);
    EXPECT_EQ(orig.woodcock.min_energy, actual.woodcock.min_energy);
    EXPECT_EQ(orig.woodcock.max_flights, actual.woodcock.max_flights);
}

//---------------------------------------------------------------------------//
/*!
 * Use a lead/argon sampling calorimeter.
 *
 * The geometry is: world > calorimeter > layer[50] > {pb, lar}.
 */
class WoodcockTest : public GeantTestBase
{
  public:
    using MevEnergy = units::MevEnergy;
    using VecStr = std::vector<std::string>;

    std::string_view gdml_basename() const override { return "testem3"; }

    GeantPhysicsOptions build_geant_options() const override
    {
        auto opts = GeantTestBase::build_geant_options();
        opts.rayleigh_scattering = true;
        return opts;
    }

    WoodcockParams::Input make_input()
    {
        auto const& core = *this->core();
        WoodcockParams::Input result;
        result.geometry = core.geometry();
        result.volume = core.volume();
        result.geomaterial = core.geomaterial();
        result.material = core.material();
        result.particle = core.particle();
        result.physics = core.physics();
        return result;
    }

    // Get a volume by name
    VolumeId find_volume(std::string const& name)
    {
        auto result = this->core()->volume()->volume_labels().find_unique(name);
        CELER_VALIDATE(result, << "no volume '" << name << "'");
        return result;
    }

    // Create one region from the named volumes
    inp::Region make_region(Label label, VecStr const& names)
    {
        inp::Region result;
        result.label = std::move(label);
        for (auto const& n : names)
        {
            result.volumes.insert(this->find_volume(n));
        }
        return result;
    }

    // Build params, using the given regions
    WoodcockParams build(std::vector<inp::Region> regions, VecStr names)
    {
        inp::WoodcockTracking options;
        options.regions = std::move(names);
        return this->build(options, std::move(regions));
    }

    WoodcockParams build(inp::WoodcockTracking const& options,
                         std::vector<inp::Region> regions)
    {
        inp::Regions r;
        r.regions = std::move(regions);
        return WoodcockParams{this->make_input(), options, r};
    }

    // Get names of root volumes
    VecStr get_roots(WoodcockParams const& params)
    {
        return this->get_volumes(
            [&data = params.host_ref()](VolumeId v) -> bool {
                return data.roots[v] != WoodcockRootId{};
            });
    }

    // Get names of volumes that can be in a root
    VecStr get_inside(WoodcockParams const& params)
    {
        return this->get_volumes(
            [&data = params.host_ref()](VolumeId v) -> bool {
                return data.inside[v];
            });
    }

    detail::MajorantBuilder make_builder(MevEnergy min_energy)
    {
        auto const& core = *this->core();
        return detail::MajorantBuilder(*core.material(),
                                       *core.particle(),
                                       *core.physics(),
                                       core.particle()->find(pdg::gamma()),
                                       min_energy);
    }

    PhysMatId find_material(std::string const& name)
    {
        auto const& mats = *this->core()->material();
        for (auto mat : range(PhysMatId{mats.size()}))
        {
            if (mats.id_to_label(mat).name == name)
            {
                return mat;
            }
        }
        CELER_VALIDATE(false, << "no material '" << name << "'");
    }

  protected:
    void SetUp() override
    {
        if constexpr (!CoreGeoTraits::has_ancestor_exit)
        {
            GTEST_SKIP() << "Not implemented for " << CoreGeoTraits::name;
        }
    }

  private:
    template<class F>
    VecStr get_volumes(F&& condition)
    {
        VecStr result;
        auto const& labels = this->core()->volume()->volume_labels();
        for (auto v : range(VolumeId{labels.size()}))
        {
            if (condition(v))
            {
                result.push_back(labels.at(v).name);
            }
        }
        return result;
    }
};

/*!
 * Test the majorant, which is independent of the geometry.
 */
class WoodcockMajorantTest : public WoodcockTest
{
  protected:
    void SetUp() override {}
};

class WoodcockUnsupportedTest : public WoodcockTest
{
  protected:
    void SetUp() override
    {
        if constexpr (CoreGeoTraits::has_ancestor_exit)
        {
            GTEST_SKIP() << "Implemented for " << CoreGeoTraits::name;
        }
    }
};

//---------------------------------------------------------------------------//

TEST_F(WoodcockUnsupportedTest, error)
{
    EXPECT_THROW(this->build({this->make_region("r", {"calorimeter"})}, {"r"}),
                 RuntimeError);
}

//---------------------------------------------------------------------------//

TEST_F(WoodcockTest, imported_region)
{
    this->core();
    auto geant_geo = global_geant_geo().lock();
    ASSERT_TRUE(geant_geo);
    auto regions = geant_geo->make_model_input().regions.regions;

    auto params = this->build(regions, {"caloregion"});
    EXPECT_EQ((VecStr{"calorimeter"}), this->get_roots(params));
    EXPECT_EQ((VecStr{"pb", "lar", "layer", "calorimeter"}),
              this->get_inside(params));
    EXPECT_EQ(1, params.host_ref().majorants.size());

    try
    {
        this->build(regions, {"caloregion", "nope"});
        FAIL() << "expected an error";
    }
    catch (RuntimeError const& e)
    {
        std::string msg{e.what()};
        EXPECT_NE(std::string::npos, msg.find("'nope'")) << msg;
        EXPECT_NE(std::string::npos, msg.find("caloregion")) << msg;
    }
}

TEST_F(WoodcockTest, root_volumes)
{
    auto const all = this->make_region("all", {"calorimeter", "layer"});
    {
        // Nested volumes not in the selected region are included
        auto params = this->build({all}, {"all"});
        EXPECT_EQ((VecStr{"calorimeter"}), this->get_roots(params));
        EXPECT_EQ((VecStr{"pb", "lar", "layer", "calorimeter"}),
                  this->get_inside(params));
    }
    {
        // Selecting a nested region separately adds a redundant root
        auto params = this->build({this->make_region("outer", {"calorimeter"}),
                                   this->make_region("inner", {"pb"})},
                                  {"outer", "inner"});
        EXPECT_EQ((VecStr{"pb", "calorimeter"}), this->get_roots(params));
        EXPECT_EQ(2, params.host_ref().majorants.size());
    }
    {
        // Regions with the same name but different extensions are merged
        auto params = this->build({this->make_region({"r", ""}, {"pb"}),
                                   this->make_region({"r", "lar"}, {"lar"}),
                                   this->make_region({"other", ""}, {"layer"})},
                                  {"r"});
        EXPECT_EQ((VecStr{"pb", "lar"}), this->get_roots(params));
        EXPECT_EQ((VecStr{"pb", "lar"}), this->get_inside(params));
    }
    {
        // World volume
        auto world = this->core()->volume()->world();
        inp::Region r;
        r.label = "world";
        r.volumes = {world};
        auto params = this->build({r}, {"world"});
        EXPECT_NE(WoodcockRootId{}, params.host_ref().roots[world]);
        EXPECT_EQ(5, this->get_inside(params).size());
    }
}

TEST_F(WoodcockTest, invalid_options)
{
    auto const region = this->make_region("r", {"calorimeter"});
    inp::WoodcockTracking options;
    options.regions = {"r"};
    {
        auto bad = options;
        bad.min_energy = MevEnergy{0};
        EXPECT_THROW(this->build(bad, {region}), RuntimeError);
    }
    {
        auto bad = options;
        bad.min_energy = MevEnergy{1e12};
        EXPECT_THROW(this->build(bad, {region}), RuntimeError);
    }
    {
        auto bad = options;
        bad.max_flights = 0;
        EXPECT_THROW(this->build(bad, {region}), RuntimeError);
    }
}

TEST_F(WoodcockMajorantTest, bounds)
{
    auto pb = this->find_material("G4_Pb");
    auto lar = this->find_material("G4_lAr");
    auto vac = this->find_material("G4_Galactic");

    // Low minimum energy includes absorption edges in the bounded range
    for (real_type min_energy : {0.001, 0.2})
    {
        auto build = this->make_builder(MevEnergy{min_energy});
        EXPECT_SOFT_EQ(1e8, value_as<MevEnergy>(build.max_energy()));
        for (auto const& mats : {std::vector{pb},
                                 std::vector{lar},
                                 std::vector{vac},
                                 std::vector{vac, lar, pb}})
        {
            auto grid = build(mats);
            double const lo = grid.x[Bound::lo];
            double const delta = (grid.x[Bound::hi] - lo) / (grid.y.size() - 1);
            EXPECT_SOFT_EQ(min_energy, std::exp(lo));

            // Check at points that are not coincident with sampled ones
            size_type num_under{0};
            double max_ratio{0};
            for (auto i : range(grid.y.size() - 1))
            {
                double const e_lo = std::exp(lo + i * delta);
                double const e_hi = std::exp(lo + (i + 1) * delta);
                for (double f :
                     {0.0, 0.033, 0.27, 0.456, 0.5, 0.61, 0.888, 0.997})
                {
                    double const energy = e_lo + f * (e_hi - e_lo);
                    double const majorant = (1 - f) * grid.y[i]
                                            + f * grid.y[i + 1];
                    double xs{0};
                    for (auto mat : mats)
                    {
                        xs = std::fmax(xs,
                                       build.calc_xs(mat, MevEnergy(energy)));
                    }
                    num_under += (majorant < xs);
                    if (energy > 0.21)
                    {
                        // Skip the first bins for checking tightness: the
                        // photoelectric cross section is discontinuous at 200
                        // keV, which may be inside the grid due to roundoff
                        max_ratio = std::fmax(max_ratio, majorant / xs);
                    }
                }
            }
            EXPECT_EQ(0, num_under);
            if (min_energy >= 0.2)
            {
                // Tight above absorption edges
                EXPECT_LT(max_ratio, 1.025);
            }
        }
    }
}

TEST_F(WoodcockMajorantTest, materials)
{
    // The majorant must not assume that the densest material has the highest
    // cross section: check that it's the maximum of the parts
    auto pb = this->find_material("G4_Pb");
    auto lar = this->find_material("G4_lAr");
    auto build = this->make_builder(MevEnergy{0.2});
    auto pb_grid = build({pb});
    auto lar_grid = build({lar});
    auto both_grid = build({lar, pb});
    ASSERT_EQ(pb_grid.y.size(), both_grid.y.size());
    for (auto i : range(both_grid.y.size()))
    {
        EXPECT_SOFT_EQ(std::fmax(pb_grid.y[i], lar_grid.y[i]), both_grid.y[i]);
    }
}

TEST_F(WoodcockTest, view)
{
    auto params = this->build({this->make_region("r", {"calorimeter"})}, {"r"});
    WoodcockView view{params.host_ref()};
    EXPECT_EQ(2, view.max_flights());

    auto gamma = this->core()->particle()->find(pdg::gamma());
    auto electron = this->core()->particle()->find(pdg::electron());
    auto lar = this->find_volume("lar");
    EXPECT_TRUE(view.is_applicable(gamma, MevEnergy{0.2}));
    EXPECT_TRUE(view.is_applicable(gamma, MevEnergy{1e6}));
    EXPECT_FALSE(view.is_applicable(gamma, MevEnergy{0.199}));
    EXPECT_FALSE(view.is_applicable(gamma, MevEnergy{1.1e8}));
    EXPECT_FALSE(view.is_applicable(electron, MevEnergy{1}));
    EXPECT_TRUE(view.can_be_inside(lar));
    EXPECT_FALSE(view.can_be_inside(this->core()->volume()->world()));
    EXPECT_FALSE(view.can_be_inside(VolumeId{}));

    // Disabled
    HostCRef<WoodcockParamsData> empty;
    EXPECT_FALSE(empty);
    EXPECT_FALSE(WoodcockView{empty}.is_applicable(gamma, MevEnergy{1}));

    // Calculate majorant
    auto build = this->make_builder(MevEnergy{0.2});
    WoodcockRootId root
        = params.host_ref().roots[this->find_volume("calorimeter")];
    ASSERT_TRUE(root);
    auto pb = this->find_material("G4_Pb");
    for (real_type e : {0.25, 1.0, 10.0, 1234.0})
    {
        auto xs = build.calc_xs(pb, MevEnergy{e});
        auto majorant = view.calc_majorant(root, MevEnergy{e});
        EXPECT_GT(majorant, xs);
        EXPECT_LT(majorant, xs * real_type(1.025));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
