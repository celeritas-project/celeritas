//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventIO.test.cc
//---------------------------------------------------------------------------//
#include "corecel/ScopedLogStorer.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/io/EventReader.hh"
#include "celeritas/io/EventWriter.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "celeritas_test.hh"
#include "celeritas_config.h"

#if CELERITAS_USE_HEPMC3
#    include <HepMC3/GenEvent.h>
#    include <HepMC3/GenParticle.h>
#    include <HepMC3/Print.h>
#    include <HepMC3/Reader.h>
#    include <HepMC3/Selector.h>
#    include <HepMC3/WriterAscii.h>
#endif

using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class EventIOTest : public Test,
                    public ::testing::WithParamInterface<char const*>
{
  protected:
    struct ReadAllResult
    {
        std::vector<int> pdg;
        std::vector<double> energy;
        std::vector<double> pos;
        std::vector<double> dir;
        std::vector<double> time;
        std::vector<int> event;
        std::vector<int> track;

        void print_expected() const;
    };

    void SetUp() override
    {
        using units::ElementaryCharge;
        using units::MevMass;
        using units::second;

        auto zero = zero_quantity();
        constexpr auto stable = ParticleRecord::stable_decay_constant();

        // Create shared standard model particle data
        particles_ = std::make_shared<ParticleParams>(ParticleParams::Input{
            {"proton",
             pdg::proton(),
             MevMass{938.27208816},
             ElementaryCharge{1},
             stable},
            {"d_quark",
             PDGNumber(1),
             MevMass{4.7},
             ElementaryCharge{-1.0 / 3},
             stable},
            {"anti_u_quark",
             PDGNumber(-2),
             MevMass{2.2},
             ElementaryCharge{-2.0 / 3},
             stable},
            {"w_minus",
             PDGNumber(-24),
             MevMass{8.0379e4},
             zero,
             1.0 / (3.157e-25 * second)},
            {"gamma", pdg::gamma(), zero, zero, stable},
        });
    }

    template<class F>
    ReadAllResult read_all(F&& read_event)
    {
        ReadAllResult result;
        std::vector<Primary> primaries;
        while (primaries = read_event(), !primaries.empty())
        {
            for (auto const& p : primaries)
            {
                result.pdg.push_back(
                    particles_->id_to_pdg(p.particle_id).unchecked_get());
                result.energy.push_back(p.energy.value());
                result.pos.insert(
                    result.pos.end(), p.position.begin(), p.position.end());
                result.dir.insert(
                    result.dir.end(), p.direction.begin(), p.direction.end());
                result.time.push_back(p.time);
                result.event.push_back(p.event_id.unchecked_get());
                result.track.push_back(p.track_id.unchecked_get());
            }
        }
        return result;
    }

    void read_write(std::string const& inp_filename,
                    std::string const& out_filename)
    {
        std::vector<Primary> primaries;
        EventReader read_event(inp_filename, particles_);
        EventWriter write_event(out_filename, particles_);
        while (primaries = read_event(), !primaries.empty())
        {
            write_event(primaries);
        }
    }

    std::shared_ptr<ParticleParams> particles_;
};

void EventIOTest::ReadAllResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static int const expected_pdg[] = "
         << repr(this->pdg)
         << ";\n"
            "EXPECT_VEC_EQ(expected_pdg, result.pdg);\n"
            "static double const expected_energy[] = "
         << repr(this->energy)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_energy, result.energy);\n"
            "static double const expected_pos[] = "
         << repr(this->pos)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);\n"
            "static double const expected_dir[] = "
         << repr(this->dir)
         << ";\n"
            "EXPECT_VEC_NEAR(expected_dir, result.dir, 1e-9);\n"
            "static double const expected_time[] = "
         << repr(this->time)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_time, result.time);\n"
            "static int const expected_event[] = "
         << repr(this->event)
         << ";\n"
            "EXPECT_VEC_EQ(expected_event, result.event);\n"
            "static int const expected_track[] = "
         << repr(this->track)
         << ";\n"
            "EXPECT_VEC_EQ(expected_track, result.track);\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_P(EventIOTest, variety_rwr)
{
    std::string const inp_filename
        = this->test_data_path("celeritas", "event-variety.hepmc3");
    std::string const ext = this->GetParam();
    std::string const out_filename
        = this->make_unique_filename(std::string{"."} + ext);

    // Read one format, write (possibly) another
    this->read_write(inp_filename, out_filename);

    // Determine the event record format and open the file
    EventReader read_event(out_filename, particles_);

    // Read events from the event record
    auto result = read_all(read_event);

    if (ext == "hepevt")
    {
        GTEST_SKIP() << "HEPEVT format sorts primaries by PDG";
    }
    // clang-format off
    static int const expected_pdg[] = {2212, 1, 2212, -2, 22, -24, 1, -2, 2212,
        1, 2212, -2, 22, -24, 1, -2, 2212, 2212, 1, -2, 22, -24, 1, -2};
    EXPECT_VEC_EQ(expected_pdg, result.pdg);
    static double const expected_energy[] = {7000000, 32238, 7000000, 57920,
        4233, 85925, 29552, 56373, 7000000, 32238, 7000000, 57920, 4233, 85925,
        29552, 56373, 7000000, 7000000, 32238, 57920, 4233, 85925, 29552,
        56373};
    EXPECT_VEC_SOFT_EQ(expected_energy, result.energy);
    static double const expected_pos[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 10, 0, 0, 10, 0, 0,
        10, 0, 0, 10, 0, 0, 10, 0, 0, 10, 0, 0, 10, 1, 2, 3, 1, 2, 3, 1, 2, 3,
        1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
    static double const expected_dir[] = {0, 0, 1, 0.023264514173899,
        -0.048669363651796, 0.99854396769596, 0, 0, -1, -0.052607942378139,
        -0.32804427475702, -0.94319635187901, -0.90094709007965,
        0.02669997932835, -0.43310674432625, 0.051894579402063,
        -0.707435663833, -0.70487001224754, -0.082735048064663,
        0.97508922087171, 0.20580554696494, 0.0702815376096, -0.87804026971226,
        -0.47339813078935, 0, 0, 1, 0.023264514173899, -0.048669363651796,
        0.99854396769596, 0, 0, -1, -0.052607942378139, -0.32804427475702,
        -0.94319635187901, -0.90094709007965, 0.02669997932835,
        -0.43310674432625, 0.051894579402063, -0.707435663833,
        -0.70487001224754, -0.082735048064663, 0.97508922087171,
        0.20580554696494, 0.0702815376096, -0.87804026971226,
        -0.47339813078935, 0, 0, 1, 0, 0, -1, 0.023264514173899,
        -0.048669363651796, 0.99854396769596, -0.052607942378139,
        -0.32804427475702, -0.94319635187901, -0.90094709007965,
        0.02669997932835, -0.43310674432625, 0.051894579402063,
        -0.707435663833, -0.70487001224754, -0.082735048064663,
        0.97508922087171, 0.20580554696494, 0.0702815376096, -0.87804026971226,
        -0.47339813078935};
    EXPECT_VEC_NEAR(expected_dir, result.dir, 1e-9);
    static double const expected_time[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1.3342563807926e-10, 1.3342563807926e-10,
        1.3342563807926e-10, 1.3342563807926e-10, 1.3342563807926e-10,
        1.3342563807926e-10, 1.3342563807926e-10, 1.3342563807926e-10};
    EXPECT_VEC_SOFT_EQ(expected_time, result.time);
    static int const expected_event[] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};
    EXPECT_VEC_EQ(expected_event, result.event);
    static int const expected_track[] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4,
        5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    EXPECT_VEC_EQ(expected_track, result.track);
    // clang-format on

    // Event reader should keep returning an empty vector
    EXPECT_TRUE(read_event().empty());
}

TEST_P(EventIOTest, no_vertex_rwr)
{
    std::string const inp_filename
        = this->test_data_path("celeritas", "event-novtx.hepmc3");
    std::string const ext = this->GetParam();
    std::string const out_filename
        = this->make_unique_filename(std::string{"."} + ext);

    // Read one format, write (possibly) another
    this->read_write(inp_filename, out_filename);

    // Read it in and check
    auto result = this->read_all(EventReader(out_filename, particles_));
    // clang-format off
    static const int expected_pdg[] = {22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
        22, 22, 22, 22, 22};
    EXPECT_VEC_EQ(expected_pdg, result.pdg);
    static const double expected_energy[] = {1000, 1000, 1000, 1000, 1000,
        1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};
    EXPECT_VEC_SOFT_EQ(expected_energy, result.energy);
    static const double expected_pos[] = {0, 0, 50, 0, 0, 50, 0, 0, 50, 0, 0,
        50, 0, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
    static const double expected_dir[] = {0.51986662880921, -0.42922054684247,
        -0.73858541172893, 0.73395459362337, 0.1872657519039, 0.65287226366497,
        -0.40053358211222, -0.081839341522929, 0.91261994925569,
        -0.51571621418069, 0.12578032404407, 0.84747631029693,
        -0.50829382271803, 0.51523183971419, -0.69005328861721,
        0.25183128898268, -0.2021612079861, -0.94642054493493,
        -0.25247976702327, 0.94617275708722, -0.20251192801867,
        0.3406634478685, -0.90517210965059, 0.25418864490188, 0.8319269271936,
        -0.54330006912643, 0.11279460402625, 0.23445050406753,
        -0.36984950110653, -0.89902408625895, 0.17562103500567,
        -0.47618127501539, 0.86163138602784, -0.60694965185736,
        0.69697036183621, 0.38189584291025, 0.51336099392838, 0.54197742792439,
        0.66537279590717, -0.36655746400947, 0.80035990702067,
        0.47440451601225, -0.7896979371307, -0.54961247309096,
        -0.27258631204511};
    EXPECT_VEC_NEAR(expected_dir, result.dir, 1e-8);
    static const double expected_time[] = {4.1028383709373e-09,
        4.1028383709373e-09, 4.1028383709373e-09, 4.1028383709373e-09,
        4.1028383709373e-09, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_VEC_SOFT_EQ(expected_time, result.time);
    static const int expected_event[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2,
        2, 2};
    EXPECT_VEC_EQ(expected_event, result.event);
    static const int expected_track[] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2,
        3, 4};
    EXPECT_VEC_EQ(expected_track, result.track);
    // clang-format on
}

TEST_P(EventIOTest, write_read)
{
    std::vector<Primary> primaries = [&particles = *this->particles_] {
        auto proton_id = particles.find(pdg::proton());
        auto gamma_id = particles.find(pdg::gamma());
        Primary gamma{gamma_id,
                      MevEnergy{1.23},
                      Real3{2, 4, 5},
                      Real3{1, 0, 0},
                      5.67e-9 * units::second,
                      EventId{0},
                      TrackId{}};
        Primary proton{proton_id,
                       MevEnergy{2.34},
                       Real3{3, 5, 8},
                       Real3{0, 1, 0},
                       5.78e-9 * units::second,
                       EventId{0},
                       TrackId{}};
        std::vector<Primary> primaries{gamma, proton, gamma, proton};
        primaries[1].position = {-3, -4, 5};
        primaries[3].position = primaries[2].position;
        primaries[3].time = primaries[2].time;
        for (auto i : range(primaries.size()))
        {
            primaries[i].track_id = TrackId(i * 2);
        }
        return primaries;
    }();

    std::string const ext = this->GetParam();
    std::string filename = this->make_unique_filename(std::string{"."} + ext);

    // Write events
    {
        EventWriter write_event(filename, particles_);
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

    // Read events
    auto result = this->read_all(EventReader(filename, particles_));

    // clang-format off
    static int const expected_pdg[] = {22, 2212, 22, 2212, 22, 2212, 22, 2212, 2212, 22};
    static double const expected_energy[] = {1.23, 2.34, 1.23, 2.34, 1.23, 2.34, 1.23, 2.34, 3.45, 1.23};
    static double const expected_pos[] = {2, 4, 5, -3, -4, 5, 2, 4, 5, 2, 4, 5, 2, 4, 5, -3, -4, 5, 2, 4, 5, 2, 4, 5, 2, 4, 5, 2, 4, 5};
    static double const expected_dir[] = {1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0};
    static double const expected_time[] = {5.67e-09, 5.78e-09, 5.67e-09, 5.67e-09, 5.67e-09, 5.78e-09, 5.67e-09, 5.67e-09, 5.67e-09, 5.67e-09};
    static int const expected_event[] = {0, 0, 0, 0, 1, 1, 1, 1, 1, 2};
    static int const expected_track[] = {0, 1, 2, 3, 0, 1, 2, 3, 4, 0};
    // clang-format on

    EXPECT_VEC_EQ(expected_pdg, result.pdg);
    EXPECT_VEC_SOFT_EQ(expected_energy, result.energy);
    EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
    EXPECT_VEC_SOFT_EQ(expected_dir, result.dir);
    EXPECT_VEC_NEAR(
        expected_time, result.time, (ext == "hepevt" ? 1e-6 : 1e-12));
    EXPECT_VEC_EQ(expected_event, result.event);
    EXPECT_VEC_EQ(expected_track, result.track);
}

INSTANTIATE_TEST_SUITE_P(EventIO,
                         EventIOTest,
                         testing::Values("hepmc3", "hepmc2", "hepevt"));

//---------------------------------------------------------------------------//
// STANDALONE TEST: HepMC3/examples/BasicExamples/basic_tree.cc
//---------------------------------------------------------------------------//

#define HepMC3Example DISABLED_StandaloneIOTest
class HepMC3Example : public Test
{
  public:
    static std::string test_filename_;
};

std::string HepMC3Example::test_filename_{};

TEST_F(HepMC3Example, write)
{
#if CELERITAS_USE_HEPMC3
    /*
     *  p1                   p7 *
     *   \                  /   *
     *    v1--p2      p5---v4   *
     *         \_v3_/       \   *
     *         /    \        p8 *
     *    v2--p4     \          *
     *   /            p6        *
     * p3                       *
     */
    using namespace HepMC3;

    GenEvent evt(Units::GEV, Units::MM);
    evt.shift_position_by(FourVector(1, 2, 3, 4));

    auto p1 = std::make_shared<GenParticle>(
        FourVector(0.0, 0.0, 7000.0, 7000.0), 2212, 3);
    auto v1 = std::make_shared<GenVertex>();
    v1->add_particle_in(p1);
    v1->set_status(4);
    evt.add_vertex(v1);

    auto p3 = std::make_shared<GenParticle>(
        FourVector(0.0, 0.0, -7000.0, 7000.0), 2212, 3);
    auto v2 = std::make_shared<GenVertex>();
    v2->add_particle_in(p3);
    evt.add_vertex(v2);

    auto p2 = std::make_shared<GenParticle>(
        FourVector(0.750, -1.569, 32.191, 32.238), 1, 3);
    v1->add_particle_out(p2);

    auto p4 = std::make_shared<GenParticle>(
        FourVector(-3.047, -19.0, -54.629, 57.920), -2, 3);
    v2->add_particle_out(p4);

    auto v3 = std::make_shared<GenVertex>();
    v3->add_particle_in(p2);
    v3->add_particle_in(p4);
    evt.add_vertex(v3);

    auto p5 = std::make_shared<GenParticle>(
        FourVector(-3.813, 0.113, -1.833, 4.233), 22, 1);
    auto p6 = std::make_shared<GenParticle>(
        FourVector(1.517, -20.68, -20.605, 85.925), -24, 3);
    v3->add_particle_out(p5);
    v3->add_particle_out(p6);

    auto v4 = std::make_shared<GenVertex>();
    v4->add_particle_in(p6);
    evt.add_vertex(v4);

    auto p7 = std::make_shared<GenParticle>(
        FourVector(-2.445, 28.816, 6.082, 29.552), 1, 1);
    auto p8 = std::make_shared<GenParticle>(
        FourVector(3.962, -49.498, -26.687, 56.373), -2, 1);
    v4->add_particle_out(p7);
    v4->add_particle_out(p8);

    Print::listing(evt);
    Print::content(evt);

    this->test_filename_ = this->make_unique_filename(".hepmc3");
    WriterAscii writer(this->test_filename_);
    writer.write_event(evt);
    writer.close();
#else
    GTEST_SKIP() << "HepMC3 is unavailable";
#endif
}

TEST_F(HepMC3Example, read)
{
#if CELERITAS_USE_HEPMC3
    using namespace HepMC3;
    ASSERT_FALSE(this->test_filename_.empty());
    Setup::set_debug_level(1);
    auto reader = open_hepmc3(this->test_filename_);

    GenEvent evt;
    reader->read_event(evt);
    Print::listing(evt);
    Print::content(evt);
#else
    GTEST_SKIP() << "HepMC3 is unavailable";
#endif
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
