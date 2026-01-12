//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventIO.test.cc
//---------------------------------------------------------------------------//
#include <fstream>

#include "corecel/Config.hh"

#include "celeritas/io/EventReader.hh"
#include "celeritas/io/EventWriter.hh"

#include "EventIOTestBase.hh"
#include "celeritas_test.hh"

#if CELERITAS_USE_HEPMC3
#    include <HepMC3/GenEvent.h>
#    include <HepMC3/GenParticle.h>
#    include <HepMC3/Print.h>
#    include <HepMC3/Reader.h>
#    include <HepMC3/WriterAscii.h>
#endif

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class EventIOTest : public EventIOTestBase,
                    public ::testing::WithParamInterface<char const*>
{
  protected:
    void read_write(std::string const& inp_filename,
                    std::string const& out_filename)
    {
        VecPrimary primaries;
        EventReader read_event(inp_filename, this->particles());
        EventWriter write_event(out_filename, this->particles());
        while (primaries = read_event(), !primaries.empty())
        {
            write_event(primaries);
        }
    }
};

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
    EventReader read_event(out_filename, this->particles());
    EXPECT_EQ(3, read_event.num_events());

    // Read events from the event record
    auto result = this->read_all(read_event);
    if (ext == "hepevt")
    {
        GTEST_SKIP() << "HEPEVT format sorts primaries by PDG";
    }

    // clang-format off
    static int const expected_pdg[] = {22, 1, -2, 22, 1, -2, 22, 1, -2};
    EXPECT_VEC_EQ(expected_pdg, result.pdg);
    static double const expected_energy[] = {4151.3789853255, 29651.503768782,
        56547.034479091, 4151.3789853255, 29651.503768782, 56547.034479091,
        4151.3789853255, 29651.503768782, 56547.034479091,};
    EXPECT_VEC_NEAR(expected_energy, result.energy, coarse_eps);
    static real_type const expected_pos[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 1, 0, 0, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3};
    EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
    static real_type const expected_dir[] = {-0.90094709007965, 0.02669997932835,
        -0.43310674432625, -0.082735048064663, 0.97508922087171,
        0.20580554696494, 0.0702815376096, -0.87804026971226,
        -0.47339813078935, -0.90094709007965, 0.02669997932835,
        -0.43310674432625, -0.082735048064663, 0.97508922087171,
        0.20580554696494, 0.0702815376096, -0.87804026971226,
        -0.47339813078935, -0.90094709007965, 0.02669997932835,
        -0.43310674432625, -0.082735048064663, 0.97508922087171,
        0.20580554696494, 0.0702815376096, -0.87804026971226,
        -0.47339813078935};
    EXPECT_VEC_NEAR(expected_dir, result.dir, coarse_eps);
    static real_type const expected_time[] = {0, 0, 0, 0, 0, 0,
        1.3342563807926e-10, 1.3342563807926e-10, 1.3342563807926e-10};
    EXPECT_VEC_SOFT_EQ(expected_time, result.time);
    static int const expected_event[] = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    EXPECT_VEC_EQ(expected_event, result.event);
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
    EventReader read_event(out_filename, this->particles());
    EXPECT_EQ(3, read_event.num_events());
    auto result = this->read_all(read_event);

    real_type energy_tol = ext == "hepevt" ? 1e-3 : coarse_eps;

    // clang-format off
    static int const expected_pdg[] = {22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
        22, 22, 22, 22, 22};
    EXPECT_VEC_EQ(expected_pdg, result.pdg);
    static real_type const expected_energy[] = {1000, 1000, 1000, 1000, 1000,
        1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};
    EXPECT_VEC_NEAR(expected_energy, result.energy, energy_tol);
    static real_type const expected_pos[] = {0, 0, 50, 0, 0, 50, 0, 0, 50, 0, 0,
        50, 0, 0, 50, 0, 0, 50, 0, 0, 50, 0, 0, 50, 0, 0,
        50, 0, 0, 50, 0, 0, 50, 0, 0, 50, 0, 0, 50, 0, 0,
        50, 0, 0, 50};
    EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
    static real_type const expected_dir[] = {0.51986662883182, -0.42922054653912,
        -0.7385854118893, 0.73395459362461, 0.18726575230281, 0.65287226354916,
        -0.40053358241289, -0.081839341451527, 0.91261994913013,
        -0.51571621404849, 0.125780323886, 0.84747631040084, -0.50829382297518,
        0.51523183959, -0.69005328852051, 0.25183128938865, -0.20216120822227,
        -0.94642054477646, -0.25247976713164, 0.94617275706344,
        -0.20251192799469, 0.34066344768752, -0.90517210955886,
        0.25418864547108, 0.83192692739206, -0.5433000688087, 0.11279460409292,
        0.23445050379268, -0.36984950141989, -0.89902408620171,
        0.17562103525404, -0.47618127524474, 0.86163138585047,
        -0.60694965222664, 0.69697036165837, 0.38189584264792,
        0.51336099422575, 0.54197742781709, 0.66537279576514,
        -0.36655746358148, 0.80035990693978, 0.47440451647941,
        -0.78969793730749, -0.54961247282688, -0.27258631206541};
    if (ext != "hepevt")
    {
        EXPECT_VEC_NEAR(expected_dir, result.dir, coarse_eps);
    }
    static real_type const expected_time[] = {4.1028383709373e-09,
        4.1028383709373e-09, 4.1028383709373e-09, 4.1028383709373e-09,
        4.1028383709373e-09, 4.1028383709373e-09,
        4.1028383709373e-09, 4.1028383709373e-09, 4.1028383709373e-09,
        4.1028383709373e-09, 4.1028383709373e-09,
        4.1028383709373e-09, 4.1028383709373e-09, 4.1028383709373e-09,
        4.1028383709373e-09};
    EXPECT_VEC_SOFT_EQ(expected_time, result.time);
    static int const expected_event[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2,
        2, 2};
    EXPECT_VEC_EQ(expected_event, result.event);
    // clang-format on
}

TEST_P(EventIOTest, write_read)
{
    std::string const ext = this->GetParam();
    std::string filename = this->make_unique_filename(std::string{"."} + ext);

    // Write events
    {
        EventWriter write_event(filename, this->particles());
        this->write_test_event(write_event);
    }

    EventReader read_event(filename, this->particles());
    if (ext == "hepevt")
    {
        // Read events but don't check
        auto result = this->read_all(read_event);

        GTEST_SKIP() << "HEPEVT results are nondeterministic";
    }
    else
    {
        this->read_check_test_event(read_event);
    }
}

TEST_P(EventIOTest, edge_case)
{
    std::string const ext = this->GetParam();
    std::string filename = this->make_unique_filename(std::string{"."} + ext);

    // Empty event file
    {
        std::ofstream out(filename);
        out << "HepMC::Version 3.02.02\n"
            << "HepMC::Asciiv3-START_EVENT_LISTING\n"
            << "HepMC::Asciiv3-END_EVENT_LISTING";
        out.close();
        EXPECT_THROW(EventReader(filename, this->particles()), RuntimeError);
    }

    // Single event
    {
        Primary p;
        p.particle_id = this->particles()->find(pdg::gamma());
        p.energy = units::MevEnergy{1};
        p.position = {0, 0, 0};
        p.direction = {1, 0, 0};
        p.event_id = EventId{0};
        std::vector<Primary> event(4, p);
        EventWriter write_event(filename, this->particles());
        write_event(event);
    }
    {
        EventReader read_event(filename, this->particles());
        EXPECT_EQ(1, read_event.num_events());
        EXPECT_EQ(4, read_event().size());
        EXPECT_TRUE(read_event().empty());
    }
}

INSTANTIATE_TEST_SUITE_P(EventIO,
                         EventIOTest,
                         testing::Values("hepmc3", "hepmc2", "hepevt"));

//---------------------------------------------------------------------------//
// STANDALONE TEST: HepMC3/examples/BasicExamples/basic_tree.cc
//---------------------------------------------------------------------------//

#define HepMC3Example DISABLED_HepMC3Example
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
