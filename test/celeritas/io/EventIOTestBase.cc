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

#include "TestMacros.hh"

using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
void EventIOTestBase::ReadAllResult::print_expected() const
{
    using std::cout;
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static int const expected_pdg[] = "
         << repr(this->pdg)
         << ";\n"
            "EXPECT_VEC_EQ(expected_pdg, result.pdg);\n"
            "static real_type const expected_energy[] = "
         << repr(this->energy)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_energy, result.energy);\n"
            "static real_type const expected_pos[] = "
         << repr(this->pos)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);\n"
            "static real_type const expected_dir[] = "
         << repr(this->dir)
         << ";\n"
            "EXPECT_VEC_NEAR(expected_dir, result.dir, 1e-8);\n"
            "static real_type const expected_time[] = "
         << repr(this->time)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_time, result.time);\n"
            "static int const expected_event[] = "
         << repr(this->event)
         << ";\n"
            "EXPECT_VEC_EQ(expected_event, result.event);\n"
            "/*** END CODE ***/\n";
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
            auto pos = to_cm(p.position);
            result.pos.insert(result.pos.end(), pos.begin(), pos.end());
            result.dir.insert(
                result.dir.end(), p.direction.begin(), p.direction.end());
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
                      EventId{0}};
        Primary proton{proton_id,
                       MevEnergy{2.34},
                       from_cm(Real3{3, 5, 8}),
                       Real3{0, 1, 0},
                       5.78e-9 * units::second,
                       EventId{0}};
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
            = {"Overwriting primary event IDs with 2: 1"};
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

    // clang-format off
    static int const expected_pdg[] = {22, 2212, 22, 2212, 22, 2212, 22, 2212,
        2212, 22};
    static real_type const expected_energy[] = {1.23, 2.34, 1.23, 2.34, 1.23,
        2.34, 1.23, 2.34, 3.45, 1.23};
    static real_type const expected_pos[] = {2, 4, 5, -3, -4, 5, 2, 4, 5, 2, 4, 5,
        2, 4, 5, -3, -4, 5, 2, 4, 5, 2, 4, 5, 2, 4, 5, 2, 4, 5};
    static real_type const expected_dir[] = {1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0,
        1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0};
    static real_type const expected_time[] = {5.67e-09, 5.78e-09, 5.67e-09,
        5.67e-09, 5.67e-09, 5.78e-09, 5.67e-09, 5.67e-09, 5.67e-09, 5.67e-09};
    static int const expected_event[] = {0, 0, 0, 0, 1, 1, 1, 1, 1, 2};
    // clang-format on

    real_type energy_tol = real_type{0.1} * coarse_eps;

    EXPECT_VEC_EQ(expected_pdg, result.pdg);
    EXPECT_VEC_NEAR(expected_energy, result.energy, energy_tol);
    EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
    EXPECT_VEC_SOFT_EQ(expected_dir, result.dir);
    EXPECT_VEC_NEAR(expected_time, result.time, real_type(1e-6));
    EXPECT_VEC_EQ(expected_event, result.event);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
