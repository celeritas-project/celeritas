//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/HepMC3PrimaryGenerator.test.cc
//---------------------------------------------------------------------------//
#include "accel/HepMC3PrimaryGenerator.hh"

#include <CLHEP/Units/SystemOfUnits.h>
#include <G4PrimaryTransformer.hh>

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/SimpleCmsTestBase.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class HepMC3PrimaryGeneratorTest : public SimpleCmsTestBase
{
  protected:
    struct ReadAllResult
    {
        std::vector<double> pos;
        std::vector<double> time;
        std::vector<int> vtx;
        std::vector<int> pdg;
        std::vector<double> energy;
        std::vector<double> dir;
        std::vector<double> mass;

        void print_expected() const;
    };

    ReadAllResult read_all(G4VPrimaryGenerator& gen, int num_events)
    {
        CELER_EXPECT(num_events > 0);
        ReadAllResult result;
        for (auto ev_id : range(num_events))
        {
            G4Event event;
            event.SetEventID(ev_id);
            EXPECT_EQ(0, event.GetNumberOfPrimaryVertex());
            gen.GeneratePrimaryVertex(&event);
            size_type num_primaries{0};
            for (auto vtx_id : range(event.GetNumberOfPrimaryVertex()))
            {
                G4PrimaryVertex* vtx = event.GetPrimaryVertex(vtx_id);
                CELER_ASSERT(vtx);
                auto pos = convert_from_geant(vtx->GetPosition(), CLHEP::cm);
                result.pos.insert(result.pos.end(), pos.begin(), pos.end());
                result.time.push_back(
                    convert_from_geant(vtx->GetT0(), CLHEP::ns));
                num_primaries += vtx->GetNumberOfParticle();
                for (auto j : range(vtx->GetNumberOfParticle()))
                {
                    G4PrimaryParticle* p = vtx->GetPrimary(j);
                    CELER_ASSERT(p);
                    result.vtx.push_back(vtx_id);
                    result.pdg.push_back(p->GetPDGcode());
                    result.energy.push_back(p->GetKineticEnergy());
                    result.mass.push_back(p->GetMass());

                    auto dir = convert_from_geant(p->GetMomentumDirection(), 1);
                    result.dir.insert(result.dir.end(), dir.begin(), dir.end());
                }
            }
            // Check that primaries are converted to Geant4 tracks without
            // warnings
            ScopedLogStorer scoped_log{&celeritas::self_logger(),
                                       LogLevel::error};
            G4PrimaryTransformer pt;
            auto* tracks = pt.GimmePrimaries(&event);
            // Primaries are added to the track vector if the \c
            // G4ParticleDefinition is not null
            EXPECT_GE(num_primaries, tracks->size());
            EXPECT_TRUE(scoped_log.empty()) << scoped_log;
        }
        return result;
    }

    void SetUp()
    {
        // Load geant4
        this->imported_data();
    }
};

void HepMC3PrimaryGeneratorTest::ReadAllResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static double const expected_pos[] = "
         << repr(this->pos)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);\n"
            "static double const expected_time[] = "
         << repr(this->time)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_time, result.time);\n"
            "static int const expected_vtx[] = "
         << repr(this->vtx)
         << ";\n"
            "EXPECT_VEC_EQ(expected_vtx, result.vtx);\n"
            "static int const expected_pdg[] = "
         << repr(this->pdg)
         << ";\n"
            "EXPECT_VEC_EQ(expected_pdg, result.pdg);\n"
            "static double const expected_energy[] = "
         << repr(this->energy)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_energy, result.energy);\n"
            "static double const expected_dir[] = "
         << repr(this->dir)
         << ";\n"
            "EXPECT_VEC_NEAR(expected_dir, result.dir, 1e-8);\n"
            "static double const expected_mass[] = "
         << repr(this->mass)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_mass, result.mass);\n"
            "/*** END CODE ***/\n";
}

TEST_F(HepMC3PrimaryGeneratorTest, no_vertex)
{
    HepMC3PrimaryGenerator generator(
        this->test_data_path("celeritas", "event-novtx.hepmc3"));
    EXPECT_EQ(3, generator.NumEvents());
    auto result = this->read_all(generator, generator.NumEvents());
    // clang-format off
    static double const expected_pos[] = {0, 0, 50, 0, 0, 50, 0, 0, 50};
    EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
    static double const expected_time[] = {4.1028383709373, 4.1028383709373, 4.1028383709373};
    EXPECT_VEC_SOFT_EQ(expected_time, result.time);
    static int const expected_vtx[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0};
    EXPECT_VEC_EQ(expected_vtx, result.vtx);
    static int const expected_pdg[] = {22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
        22, 22, 22, 22, 22};
    EXPECT_VEC_EQ(expected_pdg, result.pdg);
    static double const expected_energy[] = {1000, 1000, 1000.0000107896,
        999.99998921041, 1000, 999.99998921041, 1000.0000107896,
        999.99998921041, 999.99998921041, 1000, 1000, 999.99998921041,
        1000.0000107896, 1000.0000107896, 999.99998474121,};
    EXPECT_VEC_NEAR(expected_energy, result.energy, coarse_eps);
    static double const expected_dir[] = {0.51986662883182, -0.42922054653912,
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
    EXPECT_VEC_NEAR(expected_dir, result.dir, coarse_eps);
    static double const expected_mass[]
        = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,};
    EXPECT_VEC_EQ(expected_mass, result.mass);
    // clang-format on
}

TEST_F(HepMC3PrimaryGeneratorTest, multiple_vertex)
{
    HepMC3PrimaryGenerator generator(
        this->test_data_path("celeritas", "event-variety.hepmc3"));
    EXPECT_EQ(3, generator.NumEvents());
    auto result = this->read_all(generator, generator.NumEvents());

    // clang-format off
    static double const expected_pos[] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
        1, 2, 3, 1, 2, 3};
    EXPECT_VEC_SOFT_EQ(expected_pos, result.pos);
    static double const expected_time[] = {0, 0, 0, 0, 0.13342563807926,
        0.13342563807926};
    EXPECT_VEC_SOFT_EQ(expected_time, result.time);
    static int const expected_vtx[] = {0, 1, 1, 0, 1, 1, 0, 1, 1};
    EXPECT_VEC_EQ(expected_vtx, result.vtx);
    static int const expected_pdg[] = {22, 1, -2, 22, 1, -2, 22, 1, -2};
    EXPECT_VEC_EQ(expected_pdg, result.pdg);
    static double const expected_energy[] = {4151.3789242904, 29651.503768773,
        56547.034479342, 4151.3789242904, 29651.503768773, 56547.034479342,
        4151.3789242904, 29651.503768773, 56547.034479342,};
    EXPECT_VEC_SOFT_EQ(expected_energy, result.energy);
    static double const expected_dir[] = {-0.90094709007965, 0.02669997932835,
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
    static double const expected_mass[] = {0, -1, -1, 0, -1, -1, 0, -1, -1,};
    EXPECT_VEC_SOFT_EQ(expected_mass, result.mass);
    // clang-format on
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
