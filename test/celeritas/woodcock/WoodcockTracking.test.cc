//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/woodcock/WoodcockTracking.test.cc
//---------------------------------------------------------------------------//
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/data/AsyncSnapshot.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Stream.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"
#include "celeritas/GeantTestBase.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/geo/CoreGeoParams.hh"
#include "celeritas/geo/CoreGeoTraits.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/inp/Tracking.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/user/DetectorSteps.hh"
#include "celeritas/user/StepCollector.hh"
#include "celeritas/woodcock/WoodcockParams.hh"
#include "celeritas/woodcock/WoodcockView.hh"
#include "celeritas/woodcock/detail/MajorantBuilder.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Save the steps of primaries until they interact or leave the calorimeter.
 */
class PrimarySteps final : public StepInterface
{
  public:
    struct Step
    {
        Array<Real3, 2> pos{};
        //! Deepest volume instance, null if outside
        Array<VolumeInstanceId, 2> vol_inst{};
        Array<real_type, 2> time{};
        real_type length{};
        ActionId action{};
    };

    //! Don't filter anything
    Filters filters() const final { return {}; }

    StepSelection selection() const final
    {
        StepSelection result;
        for (auto p : range(StepPoint::size_))
        {
            result.points[p].pos = true;
            result.points[p].volume_instance_ids = true;
            result.points[p].time = true;
        }
        result.parent_id = true;
        result.event_id = true;
        result.post_step_action_id = true;
        result.step_length = true;
        return result;
    }

    // Tally host data for a step iteration
    void process_steps(HostStepState state) final
    {
        copy_steps(&steps_, state.steps);
        this->process_steps(steps_);
    }

    // Tally device data for a step iteration
    void process_steps(DeviceStepState state) final
    {
        auto& stream = device().stream(state.stream_id);
        compact_steps_async(state.steps, num_selected_);
        copy_compacted_steps_async(
            &steps_, state.steps, num_selected_.wait(), stream);
        stream.sync();
        this->process_steps(steps_);
    }

    //! Steps for a primary (one per event)
    std::vector<Step> const& steps(EventId e) const
    {
        return event_steps_[e.get()];
    }

    void resize(size_type size) { event_steps_.assign(size, {}); }

  private:
    DetectorStepOutput steps_;
    AsyncSnapshot<size_type> num_selected_;
    std::vector<std::vector<Step>> event_steps_;

    void process_steps(DetectorStepOutput const& out)
    {
        for (auto i : range(out.size()))
        {
            if (out.parent_id[i])
            {
                // Secondary
                continue;
            }
            Step s;
            for (auto p : range(StepPoint::size_))
            {
                auto const& point = out.points[p];
                s.pos[int(p)] = point.pos[i];
                s.time[int(p)] = point.time[i];
                for (auto lev : range(out.num_volume_levels))
                {
                    if (auto vi
                        = point.volume_instance_ids[i * out.num_volume_levels
                                                    + lev])
                    {
                        s.vol_inst[int(p)] = vi;
                    }
                }
            }
            s.length = out.step_length[i];
            s.action = out.post_step_action_id[i];
            event_steps_[out.event_id[i].get()].push_back(s);
        }
    }
};

//---------------------------------------------------------------------------//
/*!
 * Compare interaction locations in a sampling calorimeter against theory.
 *
 * The calorimeter is a 40 cm cube with 50 layers along x, each with 2.3 mm
 * absorber followed by 5.7 mm of another material. The default materials are
 * lead and argon.
 */
class WoodcockTrackingTest : public GeantTestBase
{
  public:
    using MevEnergy = units::MevEnergy;
    using VecDbl = std::vector<double>;

    struct Result
    {
        //! Number of interactions in each [layer][volume]
        std::vector<size_type> interactions;
        //! Number of tracks leaving the calorimeter without interacting
        size_type leaked{0};
        //! Average number of steps until interacting or leaking
        double steps_per_primary{0};
        //! Number of steps limited by each action
        std::map<std::string, size_type> actions;
    };

    static constexpr int num_layers = 50;

    std::string_view gdml_basename() const override { return "testem3"; }

    GeantPhysicsOptions build_geant_options() const override
    {
        auto opts = GeantTestBase::build_geant_options();
        opts.rayleigh_scattering = true;
        return opts;
    }

    SPConstTrackInit build_init() override
    {
        TrackInitParams::Input input;
        input.capacity = 65536;
        input.max_events = 32768;
        input.track_order = TrackOrder::none;
        return std::make_shared<TrackInitParams>(input);
    }

    SPConstWoodcock build_woodcock() override
    {
        if (!options_)
        {
            return nullptr;
        }
        WoodcockParams::Input inp;
        inp.geometry = this->geometry();
        inp.volume = this->volume();
        inp.geomaterial = this->geomaterial();
        inp.material = this->material();
        inp.particle = this->particle();
        inp.physics = this->physics();

        inp::Regions regions;
        regions.regions.resize(1);
        regions.regions.front().label = "calo";
        regions.regions.front().volumes
            = {this->volume()->volume_labels().find_unique(root_volume_)};
        return std::make_shared<WoodcockParams>(inp, options_, regions);
    }

    //! Set before constructing core
    void enable(size_type max_flights)
    {
        options_.regions = {"calo"};
        options_.max_flights = max_flights;
    }

    // Track primaries until the first interaction
    Result run(MevEnergy energy, Real3 pos, Real3 dir, size_type count);

    // Calculate probabilities of interaction in each layer
    VecDbl calc_expected(MevEnergy energy, Real3 pos, Real3 dir);

    // Calculate chi-squared and number of degrees of freedom
    std::pair<double, int> calc_chisq(std::vector<size_type> const& observed,
                                      VecDbl const& expected,
                                      size_type count) const;

    // Check the interaction distribution
    void check(Result const& actual, VecDbl const& expected, size_type count);

  protected:
    inp::WoodcockTracking options_;
    std::string root_volume_{"calorimeter"};
    //! Names of the alternating volumes and their materials
    Array<std::string, 2> volumes_{"pb", "lar"};
    Array<std::string, 2> materials_{"G4_Pb", "G4_lAr"};
    MemSpace memspace_{MemSpace::host};

    void SetUp() override
    {
        if constexpr (!CoreGeoTraits::has_ancestor_exit)
        {
            GTEST_SKIP() << "Not implemented for " << CoreGeoTraits::name;
        }
    }

  private:
    std::shared_ptr<PrimarySteps> collect_;

    std::string volume_name(VolumeInstanceId vi) const
    {
        if (!vi)
        {
            return "[outside]";
        }
        auto const& volumes = *this->core()->volume();
        return volumes.volume_labels().at(volumes.volume(vi)).name;
    }
};

//---------------------------------------------------------------------------//
auto WoodcockTrackingTest::run(
    MevEnergy energy, Real3 pos, Real3 dir, size_type count) -> Result
{
    if (!collect_)
    {
        collect_ = std::make_shared<PrimarySteps>();
        StepCollector::make_and_insert(*this->core(), {collect_});
    }
    collect_->resize(count);

    Primary p;
    p.particle_id = this->particle()->find(pdg::gamma());
    p.energy = energy;
    p.position = from_cm(pos);
    p.direction = make_unit_vector(dir);
    std::vector<Primary> primaries(count, p);
    for (auto i : range(count))
    {
        primaries[i].event_id = EventId{i};
        primaries[i].primary_id = PrimaryId{i};
    }

    StepperInput inp;
    inp.params = this->core();
    inp.stream_id = StreamId{0};
    inp.num_track_slots = count;
    inp.primary_capacity = count;
    inp.actions = std::make_shared<ActionSequence>(*this->action_reg(),
                                                   ActionSequence::Options{});
    std::unique_ptr<StepperInterface> step_ptr;
    if (memspace_ == MemSpace::host)
    {
        step_ptr = std::make_unique<Stepper<MemSpace::host>>(inp);
    }
    else
    {
        device().create_streams(1);
        step_ptr = std::make_unique<Stepper<MemSpace::device>>(inp);
    }
    StepperInterface& step = *step_ptr;

    auto const& scalars = this->physics()->host_ref().scalars;
    auto is_model = [&scalars](ActionId a) {
        return a >= scalars.first_model_action
               && a < scalars.first_model_action + scalars.num_models;
    };
    auto num_interacted = [&] {
        size_type result{0};
        for (auto i : range(EventId{count}))
        {
            auto const& steps = collect_->steps(i);
            result += std::any_of(steps.begin(), steps.end(), [&](auto& s) {
                return is_model(s.action)
                       || (s.vol_inst[0] != s.vol_inst[1]
                           && this->volume_name(s.vol_inst[0]) != "world"
                           && this->volume_name(s.vol_inst[1]) == "world");
            });
        }
        return result;
    };

    // Step until every primary has interacted or left
    step(make_span(primaries));
    size_type iters{1};
    while (num_interacted() < count)
    {
        CELER_VALIDATE(++iters < 10000, << "too many steps");
        step();
    }

    Result result;
    result.interactions.resize(2 * num_layers);
    size_type num_steps{0};
    for (auto i : range(EventId{count}))
    {
        for (auto const& s : collect_->steps(i))
        {
            ++num_steps;
            ++result.actions[std::string(
                this->action_reg()->id_to_label(s.action))];
            bool const interacted = is_model(s.action);
            bool const leaked = this->volume_name(s.vol_inst[0]) != "world"
                                && this->volume_name(s.vol_inst[1]) == "world";
            if (leaked)
            {
                ++result.leaked;
            }
            else if (interacted)
            {
                // Find layer from position
                double const x = to_cm(s.pos[1][0]) + 20;
                int const layer = static_cast<int>(std::floor(x / 0.8));
                EXPECT_TRUE(layer >= 0 && layer < num_layers) << x;
                bool const is_second = x - layer * 0.8 > 0.23;
                // Both points should be in the interaction volume
                for (auto v : s.vol_inst)
                {
                    EXPECT_EQ(volumes_[is_second], this->volume_name(v));
                }
                ++result.interactions[2 * layer + is_second];
            }
            if (interacted || leaked)
            {
                // Path should be straight
                auto delta = s.pos[1] - p.position;
                real_type const tol
                    = CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE ? 1e-6
                                                                        : 1e-4;
                EXPECT_SOFT_NEAR(real_type(1),
                                 dot_product(delta, p.direction) / norm(delta),
                                 tol);
                // Time should be consistent with distance
                EXPECT_SOFT_NEAR(
                    norm(delta) / native_value_from(units::LightSpeed{1}),
                    s.time[1],
                    tol);
                break;
            }
        }
    }
    result.steps_per_primary = double(num_steps) / count;
    return result;
}

//---------------------------------------------------------------------------//
auto WoodcockTrackingTest::calc_expected(
    MevEnergy energy, Real3 pos, Real3 dir) -> VecDbl
{
    auto const& core = *this->core();
    detail::MajorantBuilder build(*core.material(),
                                  *core.particle(),
                                  *core.physics(),
                                  core.particle()->find(pdg::gamma()),
                                  MevEnergy{0.001});
    auto calc_xs = [&](std::string const& name) -> double {
        for (auto mat : range(PhysMatId{core.material()->size()}))
        {
            if (core.material()->id_to_label(mat).name == name)
            {
                return build.calc_xs(mat, energy)
                       / native_value_from(units::InvCmXs{1});
            }
        }
        CELER_VALIDATE(false, << "no material");
    };
    Array<double, 2> const xs{calc_xs(materials_[0]), calc_xs(materials_[1])};

    dir = make_unit_vector(dir);
    CELER_VALIDATE(dir[0] > 0, << "direction must be along +x");
    // Distance until leaving the side
    double max_dist = 1000;
    for (int ax : {1, 2})
    {
        if (dir[ax] > 0)
            max_dist = std::fmin(max_dist, (20 - pos[ax]) / dir[ax]);
        else if (dir[ax] < 0)
            max_dist = std::fmin(max_dist, (-20 - pos[ax]) / dir[ax]);
    }

    VecDbl result(2 * num_layers);
    double tau{0};
    for (auto i : range(2 * num_layers))
    {
        double const lo = -20 + 0.8 * (i / 2) + (i % 2 ? 0.23 : 0);
        double const hi = -20 + 0.8 * (i / 2) + (i % 2 ? 0.8 : 0.23);
        double const start
            = std::fmin(max_dist, std::fmax(0, (lo - pos[0]) / dir[0]));
        double const stop
            = std::fmin(max_dist, std::fmax(0, (hi - pos[0]) / dir[0]));
        double const next_tau = tau + xs[i % 2] * (stop - start);
        result[i] = std::exp(-tau) - std::exp(-next_tau);
        tau = next_tau;
    }
    result.push_back(std::exp(-tau));
    return result;
}

//---------------------------------------------------------------------------//
void WoodcockTrackingTest::check(
    Result const& actual, VecDbl const& expected, size_type count)
{
    ASSERT_EQ(actual.interactions.size() + 1, expected.size());

    auto observed = actual.interactions;
    observed.push_back(actual.leaked);

    auto [chisq, ndf] = this->calc_chisq(observed, expected, count);
    ASSERT_GT(ndf, 0);
    // Four standard deviations
    EXPECT_LT(chisq, ndf + 4 * std::sqrt(2.0 * ndf));

    // Check fraction of interactions in the first material
    double obs_first{0};
    double exp_first{0};
    for (int i = 0; i < 2 * num_layers; i += 2)
    {
        obs_first += observed[i];
        exp_first += expected[i];
    }
    EXPECT_NEAR(exp_first,
                obs_first / count,
                4 * std::sqrt(exp_first * (1 - exp_first) / count));
}

//---------------------------------------------------------------------------//
std::pair<double, int> WoodcockTrackingTest::calc_chisq(
    std::vector<size_type> const& observed,
    VecDbl const& expected,
    size_type count) const
{
    CELER_EXPECT(observed.size() == expected.size());

    // Combine bins so that at least 10 counts are expected in each
    double chisq{0};
    int ndf{-1};
    double exp_accum{0};
    double obs_accum{0};
    for (auto i : range(expected.size()))
    {
        exp_accum += expected[i] * count;
        obs_accum += observed[i];
        if (exp_accum >= 10)
        {
            chisq += ipow<2>(obs_accum - exp_accum) / exp_accum;
            ++ndf;
            exp_accum = obs_accum = 0;
        }
    }
    return {chisq, ndf};
}

//---------------------------------------------------------------------------//

class WoodcockOffTest : public WoodcockTrackingTest
{
};

// Make sure the test itself is OK with surface tracking
TEST_F(WoodcockOffTest, host)
{
    size_type const count = 4000;
    Real3 const dir{1, 0, 0};
    Real3 const pos{-22, 0.01, 0.02};
    auto result = this->run(MevEnergy{1}, pos, dir, count);
    this->check(result, this->calc_expected(MevEnergy{1}, pos, dir), count);
    EXPECT_GT(result.steps_per_primary, 5);
    EXPECT_EQ(0, result.actions.count("woodcock-null"));
    EXPECT_FALSE(this->core()->host_ref().scalars.along_step_woodcock_action);
}

class WoodcockOnTest : public WoodcockTrackingTest
{
};

TEST_F(WoodcockOnTest, normal_incidence)
{
    this->enable(1000);
    size_type const count = 20000;
    Real3 const dir{1, 0, 0};
    Real3 const pos{-22, 0.01, 0.02};
    for (real_type energy : {0.21, 0.5, 1.0, 10.0, 1000.0})
    {
        SCOPED_TRACE(energy);
        auto result = this->run(MevEnergy{energy}, pos, dir, count);
        this->check(
            result, this->calc_expected(MevEnergy{energy}, pos, dir), count);

        // World to calo (one step), then reach the interaction or exit
        EXPECT_SOFT_EQ(2, result.steps_per_primary);
        EXPECT_EQ(0, result.actions.count("woodcock-null"));
    }
}

TEST_F(WoodcockOnTest, TEST_IF_CELER_DEVICE(device))
{
    memspace_ = MemSpace::device;
    this->enable(10);
    size_type const count = 20000;
    Real3 const dir{1, 0.01, 0.02};
    Real3 const pos{-22, 0.01, 0.02};
    for (real_type energy : {0.5, 5.0})
    {
        SCOPED_TRACE(energy);
        auto result = this->run(MevEnergy{energy}, pos, dir, count);
        this->check(
            result, this->calc_expected(MevEnergy{energy}, pos, dir), count);
    }
}

TEST_F(WoodcockOnTest, single_flight)
{
    this->enable(1);
    size_type const count = 4000;
    Real3 const dir{1, 0, 0};
    Real3 const pos{-22, 0.01, 0.02};
    auto result = this->run(MevEnergy{1}, pos, dir, count);
    this->check(result, this->calc_expected(MevEnergy{1}, pos, dir), count);
    EXPECT_GT(result.actions["woodcock-null"], 100);
    EXPECT_GT(result.steps_per_primary, 2);
}

TEST_F(WoodcockOnTest, side_exit)
{
    this->enable(10);
    size_type const count = 4000;
    for (Real3 dir : {Real3{1, 0, 1}, Real3{1, -2, 0.5}, Real3{1, 0.001, 0}})
    {
        SCOPED_TRACE(dir);
        Real3 const pos{-22, -15, 14};
        auto const expected = this->calc_expected(MevEnergy{2}, pos, dir);
        auto result = this->run(MevEnergy{2}, pos, dir, count);
        this->check(result, expected, count);
    }
}

TEST_F(WoodcockOnTest, start_inside)
{
    // Start inside a layer
    this->enable(10);
    size_type const count = 4000;
    Real3 const dir{1, 0.1, 0.1};
    Real3 const pos{-3.05, 1, 2};
    auto result = this->run(MevEnergy{3}, pos, dir, count);
    this->check(result, this->calc_expected(MevEnergy{3}, pos, dir), count);
    EXPECT_LT(result.steps_per_primary, 2);
}

TEST_F(WoodcockOnTest, low_energy)
{
    // Woodcock should not be applied: start in argon so that most tracks stop
    // at a boundary
    this->enable(1);
    size_type const count = 4000;
    Real3 const dir{1, 0, 0};
    Real3 const pos{-19.7, 0.01, 0.02};
    {
        auto result = this->run(MevEnergy{0.199}, pos, dir, count);
        this->check(
            result, this->calc_expected(MevEnergy{0.199}, pos, dir), count);
        EXPECT_EQ(0, result.actions.count("woodcock-null"));
        EXPECT_GT(result.actions["geo-boundary"], count / 2);
    }
    {
        // Just above the threshold
        auto result = this->run(MevEnergy{0.201}, pos, dir, count);
        this->check(
            result, this->calc_expected(MevEnergy{0.201}, pos, dir), count);
        EXPECT_GT(result.actions["woodcock-null"], 0);
        EXPECT_EQ(0, result.actions.count("geo-boundary"));
    }
}

TEST_F(WoodcockOnTest, nested_root)
{
    // Use layers as roots, so boundaries stop tracks (and the MFP has to be
    // preserved)
    root_volume_ = "layer";
    this->enable(10);
    size_type const count = 4000;
    Real3 const dir{1, 0.2, 0.3};
    Real3 const pos{-22, -5, -6};
    auto result = this->run(MevEnergy{1}, pos, dir, count);
    this->check(result, this->calc_expected(MevEnergy{1}, pos, dir), count);
    EXPECT_GT(result.actions["geo-boundary"], 2 * count);
}

TEST_F(WoodcockOnTest, sensitivity)
{
    // Make sure the statistical check would fail for a small bias: compare
    // results to the expected distribution at a slightly different energy
    this->enable(10);
    size_type const count = 20000;
    Real3 const dir{1, 0, 0};
    Real3 const pos{-22, 0.01, 0.02};
    auto result = this->run(MevEnergy{1}, pos, dir, count);
    auto expected = this->calc_expected(MevEnergy{1.1}, pos, dir);
    auto observed = result.interactions;
    observed.push_back(result.leaked);
    auto [chisq, ndf] = this->calc_chisq(observed, expected, count);
    EXPECT_GT(chisq, ndf + 4 * std::sqrt(2.0 * ndf));
}

//---------------------------------------------------------------------------//
/*!
 * Use materials for which the densest does not have the highest cross section.
 *
 * Tungsten is denser than uranium, but uranium has a higher macroscopic cross
 * section due to its atomic number.
 */
class WoodcockDenseTest : public WoodcockTrackingTest
{
  public:
    std::string_view gdml_basename() const final { return "heavy-layers"; }

  protected:
    void SetUp() override
    {
        volumes_ = {"uranium_lv", "tungsten_lv"};
        materials_ = {"uranium", "tungsten"};
        WoodcockTrackingTest::SetUp();
    }
};

TEST_F(WoodcockDenseTest, majorant)
{
    this->enable(10);
    auto const& core = *this->core();
    auto find_material = [&mats = *core.material()](std::string const& name) {
        for (auto mat : range(PhysMatId{mats.size()}))
        {
            if (mats.id_to_label(mat).name == name)
            {
                return mat;
            }
        }
        CELER_VALIDATE(false, << "no material '" << name << "'");
    };
    auto const uranium = find_material("uranium");
    auto const tungsten = find_material("tungsten");
    EXPECT_GT(core.material()->get(tungsten).density(),
              core.material()->get(uranium).density());

    auto const gamma = core.particle()->find(pdg::gamma());
    detail::MajorantBuilder build(*core.material(),
                                  *core.particle(),
                                  *core.physics(),
                                  gamma,
                                  MevEnergy{0.2});
    auto const& params = core.woodcock()->host_ref();
    WoodcockView view{params};
    WoodcockRootId const root
        = params.roots[core.volume()->volume_labels().find_unique(
            "calorimeter")];
    ASSERT_TRUE(root);

    for (real_type e : {0.21, 0.3, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 1e4, 1e6})
    {
        SCOPED_TRACE(e);
        MevEnergy const energy{e};
        auto const xs_u = build.calc_xs(uranium, energy);
        auto const xs_w = build.calc_xs(tungsten, energy);
        auto const majorant = view.calc_majorant(root, energy);
        // The majorant of the densest material would be too low
        EXPECT_GT(xs_u, xs_w);
        EXPECT_GE(majorant, xs_u);
        EXPECT_LT(majorant, xs_u * real_type(1.025));
    }
}

TEST_F(WoodcockDenseTest, interactions)
{
    this->enable(10);
    size_type const count = 20000;
    Real3 const dir{1, 0.1, 0.2};
    Real3 const pos{-22, 0.01, 0.02};
    for (real_type energy : {0.25, 1.0, 10.0})
    {
        SCOPED_TRACE(energy);
        auto result = this->run(MevEnergy{energy}, pos, dir, count);
        this->check(
            result, this->calc_expected(MevEnergy{energy}, pos, dir), count);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
