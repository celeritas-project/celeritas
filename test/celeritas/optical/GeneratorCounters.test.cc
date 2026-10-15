//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/GeneratorCounters.test.cc
//---------------------------------------------------------------------------//
#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/sys/ActionGroups.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/Device.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/Transporter.hh"
#include "celeritas/optical/gen/DirectGeneratorAction.hh"
#include "celeritas/optical/gen/GeneratorAction.hh"
#include "celeritas/optical/gen/GeneratorData.hh"
#include "celeritas/optical/gen/PrimaryGeneratorAction.hh"
#include "celeritas/phys/GeneratorInterface.hh"
#include "celeritas/phys/GeneratorRegistry.hh"
#include "celeritas/setup/StandaloneInput.hh"

#include "AssertionHelper.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;

//---------------------------------------------------------------------------//
//! Counter values for a single generator at each step
struct GenCounterResult
{
    // Current generator counters
    std::vector<size_type> buffer_size;
    std::vector<size_type> pending;
    std::vector<size_type> generated;

    // Accumulated generator counters
    std::vector<std::size_t> accum_buffer_size;
    std::vector<std::size_t> accum_pending;
    std::vector<std::size_t> accum_generated;

    //! Append the generator counters
    void append(GeneratorStateBase const& gen)
    {
        buffer_size.push_back(gen.counters.buffer_size);
        pending.push_back(gen.counters.num_pending);
        generated.push_back(gen.counters.num_generated);

        accum_buffer_size.push_back(gen.accum.buffer_size);
        accum_pending.push_back(gen.accum.num_pending);
        accum_generated.push_back(gen.accum.num_generated);
    }

    //! Print the result
    void print_expected() const
    {
        auto print = [](char const* field, auto const& values) {
            std::cout << " /* " << field << " */ " << repr(values) << ",";
        };

        print("buffer_size", buffer_size);
        print("pending", pending);
        print("generated", generated);
        print("accum_buffer_size", accum_buffer_size);
        print("accum_pending", accum_pending);
        print("accum_generated", accum_generated);
    }
};

//! Core counters and counters for each optical generator
struct CounterResult
{
    // Core counters
    std::vector<size_type> pending;
    std::vector<size_type> generated;
    std::vector<size_type> initializers;
    std::vector<size_type> vacancies;
    std::vector<size_type> active;
    std::vector<size_type> secondaries;
    std::vector<size_type> alive;
    std::vector<size_type> cut;
    std::vector<size_type> errored;

    // Counters for each generator
    std::map<std::string, GenCounterResult> gen;

    //! Append the core counters
    void append(CoreStateCounters const& core)
    {
        pending.push_back(core.num_pending);
        generated.push_back(core.num_generated);
        initializers.push_back(core.num_initializers);
        vacancies.push_back(core.num_vacancies);
        active.push_back(core.num_active);
        secondaries.push_back(core.num_secondaries);
        alive.push_back(core.num_alive);
        cut.push_back(core.num_cut);
        errored.push_back(core.num_errored);
    }

    //! Print the result
    void print_expected() const
    {
        auto print = [](char const* field, auto const& values) {
            std::cout << " /* " << field << " */ " << repr(values) << ",";
        };

        std::cout << "CounterResult const expected_counts{";
        print("pending", pending);
        print("generated", generated);
        print("initializers", initializers);
        print("vacancies", vacancies);
        print("active", active);
        print("secondaries", secondaries);
        print("alive", alive);
        print("cut", cut);
        print("errored", errored);
        std::cout << " /* gen */ {";
        for (auto const& [gen_label, gen_result] : gen)
        {
            std::cout << "{\"" << gen_label << "\",{";
            gen_result.print_expected();
            std::cout << "}},";
        }
        std::cout << "},};\n";
    }
};

//! Compare the counters and report where they differ
::testing::AssertionResult IsRefEq(char const* expr1,
                                   char const* expr2,
                                   CounterResult const& val1,
                                   CounterResult const& val2)
{
    AssertionHelper result{expr1, expr2};

    auto check = [&result](std::string const& name,
                           auto const& expected,
                           auto const& actual) {
        if (auto r = ::celeritas::testdetail::IsVecEq(
                name.c_str(), name.c_str(), expected, actual);
            !static_cast<bool>(r))
        {
            result.fail() << "\n" << name << ": " << r.message();
        }
    };

    // Check the core counters
    check("pending", val1.pending, val2.pending);
    check("generated", val1.generated, val2.generated);
    check("initializers", val1.initializers, val2.initializers);
    check("vacancies", val1.vacancies, val2.vacancies);
    check("active", val1.active, val2.active);
    check("secondaries", val1.secondaries, val2.secondaries);
    check("alive", val1.alive, val2.alive);
    check("cut", val1.cut, val2.cut);
    check("errored", val1.errored, val2.errored);

    // Check that the same generators are present
    std::vector<std::string> labels1;
    std::vector<std::string> labels2;
    for (auto const& kv : val1.gen)
    {
        labels1.push_back(kv.first);
    }
    for (auto const& kv : val2.gen)
    {
        labels2.push_back(kv.first);
    }
    check("gen", labels1, labels2);

    // Check the counters for each generator
    for (auto const& [label, gen1] : val1.gen)
    {
        auto iter = val2.gen.find(label);
        if (iter == val2.gen.end())
        {
            continue;
        }
        auto const& gen2 = iter->second;
        check(label + ".buffer_size", gen1.buffer_size, gen2.buffer_size);
        check(label + ".pending", gen1.pending, gen2.pending);
        check(label + ".generated", gen1.generated, gen2.generated);
        check(label + ".accum_buffer_size",
              gen1.accum_buffer_size,
              gen2.accum_buffer_size);
        check(label + ".accum_pending", gen1.accum_pending, gen2.accum_pending);
        check(label + ".accum_generated",
              gen1.accum_generated,
              gen2.accum_generated);
    }

    return result;
}

//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//
class GeneratorCountersTestBase : public Test
{
  public:
    //!@{
    //! \name Type aliases
    using GT = GeneratorType;
    using Speed = units::LightSpeed;
    using CoreStateHost = CoreState<MemSpace::host>;
    using CoreStateDevice = CoreState<MemSpace::device>;
    //!@}

  protected:
    //! Get an identifying key for the geometry (basename, description, etc)
    virtual std::string gdml_basename() const = 0;

    void SetUp() override
    {
        // Set geometry filename
        osi_.problem.model.geometry
            = Test::test_data_path("geocel", this->gdml_basename() + ".gdml");

        // Set per-process state sizes
        osi_.problem.num_streams = 1;
        osi_.problem.capacity.tracks = 8;
        osi_.problem.capacity.primaries = 64;
        osi_.problem.capacity.generators = 4;

        // No WLS: the only pending photons are inserted by this test.
        osi_.geant_setup.cherenkov = std::nullopt;
        osi_.geant_setup.scintillation = std::nullopt;
        osi_.geant_setup.wavelength_shifting = std::nullopt;
        osi_.geant_setup.wavelength_shifting2 = std::nullopt;
        osi_.geant_setup.rayleigh_scattering = false;
        osi_.geant_setup.mie_scattering = false;
    }

    //! Create the host or device state
    std::shared_ptr<CoreStateBase> make_state(CoreParams const&,
                                              size_type) const;

    //! Run the stepping loop and append counters at each step
    CounterResult run(CoreParams const&, CoreStateBase&) const;

    inp::OpticalStandaloneInput osi_;
};

class LarSphereGeneratorCountersTest : public GeneratorCountersTestBase
{
  public:
    std::string gdml_basename() const final { return "lar-sphere"; }
};

class WlsSlabGeneratorCountersTest : public GeneratorCountersTestBase
{
  public:
    std::string gdml_basename() const final { return "wls-slab"; }
};

//---------------------------------------------------------------------------//
/*!
 * Construct the host or device state.
 */
std::shared_ptr<CoreStateBase> GeneratorCountersTestBase::make_state(
    CoreParams const& params, size_type num_track_slots) const
{
    std::shared_ptr<CoreStateBase> state;
    MemSpace memspace = celeritas::device() ? MemSpace::device : MemSpace::host;

    if (memspace == MemSpace::device)
    {
        state = std::make_shared<CoreState<MemSpace::device>>(
            params, StreamId{0}, num_track_slots);
    }
    else
    {
        state = std::make_shared<CoreState<MemSpace::host>>(
            params, StreamId{0}, num_track_slots);
    }

    state->aux() = std::make_shared<AuxStateVec>(
        *params.aux_reg(), memspace, state->stream_id(), state->size());
    return state;
}

//---------------------------------------------------------------------------//
/*!
 * Run the stepping loop and append counters at each step.
 */
CounterResult GeneratorCountersTestBase::run(CoreParams const& params,
                                             CoreStateBase& state) const
{
    CounterResult result;
    auto const& gen_reg = *params.gen_reg();

    // Append the core counters and each generator's counters
    auto append_all = [&](CoreStateCounters const& counters) {
        result.append(counters);
        for (auto id : range(GeneratorId{gen_reg.size()}))
        {
            result.gen[gen_reg.id_to_label(id)].append(
                gen_reg.at(id)->counters(*state.aux()));
        }
    };

    // Append the counters before starting the stepping loop
    auto counters = state.sync_get_counters();
    append_all(counters);

    ActionGroups<CoreParams, CoreState> actions(*params.action_reg());

    while (counters.num_pending > 0 || counters.num_alive > 0)
    {
        for (auto const& action : actions.step())
        {
            if (auto* s = dynamic_cast<CoreStateHost*>(&state))
            {
                action->step(params, *s);
            }
            else if (auto* s = dynamic_cast<CoreStateDevice*>(&state))
            {
                action->step(params, *s);
            }
            else
            {
                CELER_ASSERT_UNREACHABLE();
            }
        }
        // Append counts at the end of the step
        counters = state.sync_get_counters();
        append_all(counters);
    }
    return result;
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(LarSphereGeneratorCountersTest, offload)
{
    constexpr size_type num_track_slots = 32;

    // Enable Cherenkov and scintillation generation
    osi_.problem.generator = inp::OpticalOffloadGenerator{};
    osi_.geant_setup.cherenkov.emplace();
    osi_.geant_setup.scintillation.emplace();

    // Construct the params and state
    auto loaded = setup::standalone_input(osi_);
    auto const& params = *loaded.problem.transporter->params();
    auto state = this->make_state(params, num_track_slots);

    // Create and insert optical distribution data.
    std::vector<size_type> photon_counts{{117, 19, 2, 49, 13}};
    auto distributions = [&photon_counts] {
        std::vector<GT> types{GT::cherenkov, GT::scintillation};

        GeneratorDistributionData data;
        data.step_length = from_cm(0.2);
        data.charge = units::ElementaryCharge{-1};
        data.material = OptMatId{0};
        data.continuous_edep_fraction = 1;
        data.points[StepPoint::pre] = {Speed{0.7}, 0, from_cm(Real3{0, 0, 0})};
        data.points[StepPoint::post]
            = {Speed{0.6}, 1e-11, from_cm(Real3{0, 0, 0.2})};

        std::vector<GeneratorDistributionData> result(photon_counts.size(),
                                                      data);
        for (auto i : range(photon_counts.size()))
        {
            result[i].type = types[i % types.size()];
            result[i].num_photons = photon_counts[i];
            CELER_ASSERT(result[i]);
        }
        return result;
    }();
    auto generator
        = std::dynamic_pointer_cast<GeneratorAction>(loaded.problem.generator);
    ASSERT_TRUE(generator);
    generator->insert(*state, make_span(distributions));
    // TODO: move this to generator action?
    size_type const num_photons
        = std::accumulate(photon_counts.begin(), photon_counts.end(), 0);
    state->add_pending(num_photons);

    auto counts = this->run(params, *state);
    // counts.print_expected();

    CounterResult const expected_counts{
        /* pending */ {200ul, 168ul, 137ul, 106ul, 74ul, 46ul, 16ul, 0ul},
        /* generated */ {0ul, 32ul, 63ul, 94ul, 126ul, 154ul, 184ul, 200ul},
        /* initializers */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* vacancies */ {32ul, 31ul, 31ul, 32ul, 28ul, 30ul, 31ul, 32ul},
        /* active */ {0ul, 32ul, 32ul, 32ul, 32ul, 32ul, 32ul, 17ul},
        /* secondaries */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* alive */ {0ul, 1ul, 1ul, 0ul, 4ul, 2ul, 1ul, 0ul},
        /* cut */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* errored */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* gen */
        {
            {"generate",
             {
                 /* buffer_size */ {5ul, 5ul, 5ul, 5ul, 5ul, 5ul, 5ul, 0ul},
                 /* pending */ {0ul, 168ul, 137ul, 106ul, 74ul, 46ul, 16ul, 0ul},
                 /* generated */ {0ul, 32ul, 63ul, 94ul, 126ul, 154ul, 184ul, 0ul},
                 /* accum_buffer_size */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 5ul},
                 /* accum_pending */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
                 /* accum_generated */
                 {0ul, 32ul, 63ul, 94ul, 126ul, 154ul, 184ul, 200ul},
             }},
        },
    };
    EXPECT_REF_EQ(expected_counts, counts);
}

TEST_F(WlsSlabGeneratorCountersTest, primary)
{
    constexpr size_type num_track_slots = 32;

    // Create primary generator input with WLS enabled
    osi_.geant_setup.wavelength_shifting.emplace();
    osi_.geant_setup.wavelength_shifting2.emplace();
    osi_.problem.capacity.generators = 32;
    osi_.problem.generator = [] {
        inp::OpticalPrimaryGenerator gen;
        gen.primaries = 49;
        gen.energy = inp::MonoenergeticDistribution{9.5e-6};
        gen.angle = inp::IsotropicDistribution{};
        gen.shape = inp::PointDistribution{{0, 0, 0}};
        return gen;
    }();

    // Construct the params and state
    auto loaded = setup::standalone_input(osi_);
    auto const& params = *loaded.problem.transporter->params();
    auto state = this->make_state(params, num_track_slots);

    // Create and insert optical distribution data.
    auto generator = std::dynamic_pointer_cast<PrimaryGeneratorAction>(
        loaded.problem.generator);
    ASSERT_TRUE(generator);
    generator->insert(*state);

    auto counts = this->run(params, *state);
    // counts.print_expected();

    CounterResult const expected_counts{
        /* pending */ {49ul, 38ul, 23ul, 11ul, 2ul, 2ul, 1ul, 0ul},
        /* generated */ {0ul, 32ul, 64ul, 87ul, 98ul, 100ul, 102ul, 103ul},
        /* initializers */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* vacancies */ {32ul, 32ul, 32ul, 32ul, 32ul, 32ul, 32ul, 32ul},
        /* active */ {0ul, 32ul, 32ul, 23ul, 11ul, 2ul, 2ul, 1ul},
        /* secondaries */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* alive */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* cut */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* errored */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* gen */
        {
            {"primary-generate",
             {
                 /* buffer_size */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
                 /* pending */ {49ul, 17ul, 6ul, 0ul, 0ul, 0ul, 0ul, 0ul},
                 /* generated */ {0ul, 32ul, 43ul, 49ul, 49ul, 49ul, 49ul, 49ul},
                 /* accum_buffer_size */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
                 /* accum_pending */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
                 /* accum_generated */ {0ul, 32ul, 43ul, 49ul, 49ul, 49ul, 49ul, 49ul},
             }},
            {"wls-generate",
             {
                 /* buffer_size */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
                 /* pending */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
                 /* generated */ {0ul, 0ul, 21ul, 38ul, 49ul, 51ul, 53ul, 54ul},
                 /* accum_buffer_size */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
                 /* accum_pending */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
                 /* accum_generated */ {0ul, 0ul, 21ul, 38ul, 49ul, 51ul, 53ul, 54ul},
             }},
        },
    };
    EXPECT_REF_EQ(expected_counts, counts);
}

TEST_F(LarSphereGeneratorCountersTest, direct)
{
    constexpr size_type num_track_slots = 32;

    // Create direct generator input
    osi_.geant_setup.rayleigh_scattering = true;
    osi_.problem.generator = inp::OpticalDirectGenerator{};

    // Limit the number of steps per track so that some tracks are cut
    osi_.problem.limits.steps = 4;

    std::vector<TrackInitializer> const inits(
        89,
        optical::TrackInitializer{units::MevEnergy{9.5e-6},
                                  Real3{0, 0, 0},  // position
                                  Real3{1, 0, 0},  // direction
                                  Real3{0, 1, 0},  // polarization
                                  0,  // time
                                  {},  // primary
                                  ImplVolumeId{0}});

    // Construct the params and state
    auto loaded = setup::standalone_input(osi_);
    auto const& params = *loaded.problem.transporter->params();
    auto state = this->make_state(params, num_track_slots);

    // Create and insert optical distribution data.
    auto generator = std::dynamic_pointer_cast<DirectGeneratorAction>(
        loaded.problem.generator);
    ASSERT_TRUE(generator);
    generator->insert(*state, make_span(inits));

    // Check direct generator counters
    auto counts = this->run(params, *state);
    // counts.print_expected();

    CounterResult const expected_counts{
        /* pending */ {89ul, 57ul, 37ul, 16ul, 0ul, 0ul, 0ul},
        /* generated */ {0ul, 32ul, 52ul, 73ul, 89ul, 89ul, 89ul},
        /* initializers */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* vacancies */ {32ul, 20ul, 21ul, 23ul, 24ul, 26ul, 32ul},
        /* active */ {0ul, 32ul, 32ul, 32ul, 25ul, 8ul, 6ul},
        /* secondaries */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* alive */ {0ul, 12ul, 11ul, 9ul, 8ul, 6ul, 0ul},
        /* cut */ {0ul, 0ul, 0ul, 0ul, 2ul, 2ul, 3ul},
        /* errored */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* gen */
        {
            {"generate-direct",
             {
                 /* buffer_size */ {89ul, 89ul, 89ul, 89ul, 0ul, 0ul, 0ul},
                 /* pending */ {89ul, 57ul, 37ul, 16ul, 0ul, 0ul, 0ul},
                 /* generated */ {0ul, 32ul, 52ul, 73ul, 0ul, 0ul, 0ul},
                 /* accum_buffer_size */ {0ul, 0ul, 0ul, 0ul, 89ul, 89ul, 89ul},
                 /* accum_pending */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
                 /* accum_generated */ {0ul, 32ul, 52ul, 73ul, 89ul, 89ul, 89ul},
             }},
        },
    };
    EXPECT_REF_EQ(expected_counts, counts);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
