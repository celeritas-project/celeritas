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
#include "corecel/sys/Device.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
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
#include "Comparer.hh"
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

    //// REFERENCE RESULT ////

    static constexpr char class_str[] = "GenCounterResult";

    friend std::ostream& operator<<(std::ostream& os,
                                    GenCounterResult const& ref)
    {
        return os << '{' << CELER_ATTR_REPR(ref, buffer_size) << ", "
                  << CELER_ATTR_REPR(ref, pending) << ", "
                  << CELER_ATTR_REPR(ref, generated) << ", "
                  << CELER_ATTR_REPR(ref, accum_buffer_size) << ", "
                  << CELER_ATTR_REPR(ref, accum_pending) << ", "
                  << CELER_ATTR_REPR(ref, accum_generated) << '}';
    }

    friend bool check_ref_eq(AssertionHelper& helper,
                             GenCounterResult const& expected,
                             GenCounterResult const& actual,
                             std::nullptr_t = {})
    {
        RefComparer<GenCounterResult> compare{helper, expected, actual};
        CELER_INVOKE_ATTR(compare.is_eq, GenCounterResult, buffer_size);
        CELER_INVOKE_ATTR(compare.is_eq, GenCounterResult, pending);
        CELER_INVOKE_ATTR(compare.is_eq, GenCounterResult, generated);
        CELER_INVOKE_ATTR(compare.is_eq, GenCounterResult, accum_buffer_size);
        CELER_INVOKE_ATTR(compare.is_eq, GenCounterResult, accum_pending);
        CELER_INVOKE_ATTR(compare.is_eq, GenCounterResult, accum_generated);
        return static_cast<bool>(compare);
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

    //// REFERENCE RESULT ////

    static constexpr char class_str[] = "CounterResult";

    friend std::ostream& operator<<(std::ostream& os, CounterResult const& ref)
    {
        os << '{' << CELER_ATTR_REPR(ref, pending) << ", "
           << CELER_ATTR_REPR(ref, generated) << ", "
           << CELER_ATTR_REPR(ref, initializers) << ", "
           << CELER_ATTR_REPR(ref, vacancies) << ", "
           << CELER_ATTR_REPR(ref, active) << ", "
           << CELER_ATTR_REPR(ref, secondaries) << ", "
           << CELER_ATTR_REPR(ref, alive) << ", " << CELER_ATTR_REPR(ref, cut)
           << ", " << CELER_ATTR_REPR(ref, errored) << ", "
           << "/* gen = */ {";
        for (auto const& [label, gen_result] : ref.gen)
        {
            os << '{' << repr(label) << ", " << gen_result << "},";
        }
        return os << "}}";
    }

    friend bool check_ref_eq(AssertionHelper& helper,
                             CounterResult const& expected,
                             CounterResult const& actual,
                             std::nullptr_t = {})
    {
        RefComparer<CounterResult> compare{helper, expected, actual};
        CELER_INVOKE_ATTR(compare.is_eq, CounterResult, pending);
        CELER_INVOKE_ATTR(compare.is_eq, CounterResult, generated);
        CELER_INVOKE_ATTR(compare.is_eq, CounterResult, initializers);
        CELER_INVOKE_ATTR(compare.is_eq, CounterResult, vacancies);
        CELER_INVOKE_ATTR(compare.is_eq, CounterResult, active);
        CELER_INVOKE_ATTR(compare.is_eq, CounterResult, secondaries);
        CELER_INVOKE_ATTR(compare.is_eq, CounterResult, alive);
        CELER_INVOKE_ATTR(compare.is_eq, CounterResult, cut);
        CELER_INVOKE_ATTR(compare.is_eq, CounterResult, errored);

        std::vector<std::string> expected_labels;
        std::vector<std::string> actual_labels;
        for (auto const& item : expected.gen)
        {
            expected_labels.push_back(item.first);
        }
        for (auto const& item : actual.gen)
        {
            actual_labels.push_back(item.first);
        }
        compare.record_result(
            is_vec_eq(helper, "gen", expected_labels, actual_labels));

        for (auto const& [label, expected_gen] : expected.gen)
        {
            auto iter = actual.gen.find(label);
            if (iter != actual.gen.end())
            {
                bool const gen_success
                    = check_ref_eq(helper, expected_gen, iter->second);
                compare.record_result(gen_success);
                if (!gen_success)
                {
                    helper.fail() << "...in gen[\"" << label << "\"]";
                }
            }
            else
            {
                helper.fail() << "Missing label " << label;
                compare.record_result(false);
            }
        }
        return static_cast<bool>(compare);
    }
};

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

    auto counts = this->run(params, *state);
    // print_expected(counts);

    CounterResult const ref{
        /* pending = */ {200ul, 168ul, 137ul, 106ul, 74ul, 46ul, 16ul, 0ul},
        /* generated = */ {0ul, 32ul, 63ul, 94ul, 126ul, 154ul, 184ul, 200ul},
        /* initializers = */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* vacancies = */ {32ul, 31ul, 31ul, 32ul, 28ul, 30ul, 31ul, 32ul},
        /* active = */ {0ul, 32ul, 32ul, 32ul, 32ul, 32ul, 32ul, 17ul},
        /* secondaries = */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* alive = */ {0ul, 1ul, 1ul, 0ul, 4ul, 2ul, 1ul, 0ul},
        /* cut = */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* errored = */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
        /* gen = */
        {
            {"generate",
             {
                 /* buffer_size = */ {5ul, 5ul, 5ul, 5ul, 5ul, 5ul, 5ul, 0ul},
                 /* pending = */ {0ul, 168ul, 137ul, 106ul, 74ul, 46ul, 16ul, 0ul},
                 /* generated = */ {0ul, 32ul, 63ul, 94ul, 126ul, 154ul, 184ul, 200ul},
                 /* accum_buffer_size = */ {0ul, 5ul, 5ul, 5ul, 5ul, 5ul, 5ul, 5ul},
                 /* accum_pending = */
                 {0ul, 200ul, 200ul, 200ul, 200ul, 200ul, 200ul, 200ul},
                 /* accum_generated = */
                 {0ul, 32ul, 63ul, 94ul, 126ul, 154ul, 184ul, 200ul},
             }},
        },
    };
    if (Test::is_reference_rng)
    {
        EXPECT_REF_EQ(ref, counts);
    }
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
                 /* accum_pending */ {49ul, 49ul, 49ul, 49ul, 49ul, 49ul, 49ul, 49ul},
                 /* accum_generated */ {0ul, 32ul, 43ul, 49ul, 49ul, 49ul, 49ul, 49ul},
             }},
            {"wls-generate",
             {
                 /* buffer_size */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
                 /* pending */ {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul},
                 /* generated */ {0ul, 0ul, 21ul, 38ul, 49ul, 51ul, 53ul, 54ul},
                 /* accum_buffer_size */ {0ul, 0ul, 14ul, 25ul, 35ul, 36ul, 38ul, 39ul},
                 /* accum_pending */ {0ul, 0ul, 21ul, 38ul, 49ul, 51ul, 53ul, 54ul},
                 /* accum_generated */ {0ul, 0ul, 21ul, 38ul, 49ul, 51ul, 53ul, 54ul},
             }},
        },
    };
    if (Test::is_reference_rng)
    {
        EXPECT_REF_EQ(expected_counts, counts);
    }
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
                 /* generated */ {0ul, 32ul, 52ul, 73ul, 89ul, 89ul, 89ul},
                 /* accum_buffer_size */ {89ul, 89ul, 89ul, 89ul, 89ul, 89ul, 89ul},
                 /* accum_pending */ {89ul, 89ul, 89ul, 89ul, 89ul, 89ul, 89ul},
                 /* accum_generated */ {0ul, 32ul, 52ul, 73ul, 89ul, 89ul, 89ul},
             }},
        },
    };
    if (Test::is_reference_rng)
    {
        EXPECT_REF_EQ(expected_counts, counts);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
