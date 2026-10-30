//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventIOTestBase.cc
//---------------------------------------------------------------------------//
#include "EventIOTestBase.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/Repr.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "TestMacros.hh"

using namespace celeritas::literals;
using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//

std::ostream& operator<<(std::ostream& os,
                         EventIOTestBase::ReadAllResult const& ref)
{
    return os << '{' << CELER_ATTR_REPR(ref, pdg) << ", "
              << CELER_ATTR_REPR(ref, energy) << ", "
              << CELER_ATTR_REPR(ref, pos) << ", " << CELER_ATTR_REPR(ref, dir)
              << ", " << CELER_ATTR_REPR(ref, time) << ", "
              << CELER_ATTR_REPR(ref, event) << '}';
}

void check_ref_eq(AssertionHelper& helper,
                  EventIOTestBase::ReadAllResult const& expected,
                  EventIOTestBase::ReadAllResult const& actual,
                  EventIOTestBase::ReadAllResult::Tolerance const& tol)
{
    using namespace ::celeritas::test;
    is_vec_eq(helper, "pdg", expected.pdg, actual.pdg);
    is_vec_soft_eq(
        helper, "energy", expected.energy, actual.energy, tol.energy);
    is_vec_soft_eq(helper, "pos", expected.pos, actual.pos, tol.pos);
    is_vec_soft_eq(helper, "dir", expected.dir, actual.dir, tol.dir);
    is_vec_soft_eq(helper, "time", expected.time, actual.time, tol.time);
    is_vec_eq(helper, "event", expected.event, actual.event);
}

//---------------------------------------------------------------------------//

void EventIOTestBase::SetUp()
{
    using namespace constants;
    using units::ElementaryCharge;
    using units::MevMass;
    using units::second;

    auto zero = zero_quantity();

    // Create shared standard model particle data
    particles_ = std::make_shared<ParticleParams>(ParticleParams::Input{
        {"proton",
         pdg::proton(),
         MevMass{938.27208816},
         ElementaryCharge{1},
         stable_decay_constant},
        {"d_quark",
         PDGNumber(1),
         MevMass{4.7},
         ElementaryCharge{-1.0 / 3},
         stable_decay_constant},
        {"anti_u_quark",
         PDGNumber(-2),
         MevMass{2.2},
         ElementaryCharge{-2.0 / 3},
         stable_decay_constant},
        {"w_minus",
         PDGNumber(-24),
         MevMass{8.0379e4},
         zero,
         1.0 / (3.157e-25 * second)},
        {"gamma", pdg::gamma(), zero, zero, stable_decay_constant},
    });
}

//---------------------------------------------------------------------------//
/*!
 * Read all primaries from a file.
 */
auto EventIOTestBase::read_all(Reader& read_event) const -> ReadAllResult
{
    ReadAllResult result;
    VecPrimary primaries;
    while (primaries = read_event(), !primaries.empty())
    {
        for (auto const& p : primaries)
        {
            result.pdg.push_back(
                particles_->id_to_pdg(p.particle_id).unchecked_get());
            result.energy.push_back(p.energy.value());
            result.pos.push_back(to_cm(p.position));
            result.dir.push_back(p.direction);
            result.time.push_back(p.time / units::second);
            result.event.push_back(p.event_id.unchecked_get());
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Write several events.
 */
void EventIOTestBase::write_test_event(Writer& write_event) const
{
    std::vector<Primary> primaries = [&particles = *this->particles_] {
        auto proton_id = particles.find(pdg::proton());
        auto gamma_id = particles.find(pdg::gamma());
        Primary gamma{gamma_id,
                      MevEnergy{1.23},
                      from_cm(Real3{2, 4, 5}),
                      Real3{1, 0, 0},
                      5.67e-9 * units::second,
                      /* weight = */ 1.0,
                      EventId{0},
                      PrimaryId{0},
                      UniqueEventId{}};
        Primary proton{proton_id,
                       MevEnergy{2.34},
                       from_cm(Real3{3, 5, 8}),
                       Real3{0, 1, 0},
                       5.78e-9 * units::second,
                       /* weight = */ 1.0,
                       EventId{0},
                       PrimaryId{0},
                       UniqueEventId{}};
        std::vector<Primary> primaries{gamma, proton, gamma, proton};
        primaries[1].position = from_cm(Real3{-3, -4, 5});
        primaries[3].position = primaries[2].position;
        primaries[3].time = primaries[2].time;
        return primaries;
    }();

    write_event(primaries);

    // Add another primary and update event ID
    primaries.push_back(primaries.back());
    primaries.back().energy = MevEnergy{3.45};
    for (auto& p : primaries)
    {
        p.event_id = EventId{1};
    }
    write_event(primaries);

    // Write a single primary with incorrect ID
    primaries.erase(primaries.begin() + 1, primaries.end());
    {
        ScopedLogStorer scoped_log_{&celeritas::self_logger()};
        write_event(primaries);
        static char const* const expected_log_messages[]
            = {"Event IDs will not match output: this is a known issue"};
        EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
        static char const* const expected_log_levels[] = {"warning"};
        EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Read and check the test primaries from a file.
 */
void EventIOTestBase::read_check_test_event(Reader& read_event) const
{
    auto result = this->read_all(read_event);

    ReadAllResult ref;
    ref.pdg = {22, 2212, 22, 2212, 22, 2212, 22, 2212, 2212, 22};
    ref.energy = {1.23, 2.34, 1.23, 2.34, 1.23, 2.34, 1.23, 2.34, 3.45, 1.23};
    ref.pos = {{2, 4, 5},
               {-3, -4, 5},
               {2, 4, 5},
               {2, 4, 5},
               {2, 4, 5},
               {-3, -4, 5},
               {2, 4, 5},
               {2, 4, 5},
               {2, 4, 5},
               {2, 4, 5}};
    ref.dir = {{1, 0, 0},
               {0, 1, 0},
               {1, 0, 0},
               {0, 1, 0},
               {1, 0, 0},
               {0, 1, 0},
               {1, 0, 0},
               {0, 1, 0},
               {0, 1, 0},
               {1, 0, 0}};
    ref.time = {5.67e-09,
                5.78e-09,
                5.67e-09,
                5.67e-09,
                5.67e-09,
                5.78e-09,
                5.67e-09,
                5.67e-09,
                5.67e-09,
                5.67e-09};
    ref.event = {0, 0, 0, 0, 1, 1, 1, 1, 1, 2};

    real_type energy_tol = 0.1_r * coarse_eps;
    ReadAllResult::Tolerance tol;
    tol.energy = energy_tol;
    tol.time = 1e-6_r;
    EXPECT_REF_NEAR(ref, result, tol);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
