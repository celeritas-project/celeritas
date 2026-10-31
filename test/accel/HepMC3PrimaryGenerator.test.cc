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
#include "geocel/GenericGeoResults.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/SimpleCmsTestBase.hh"

#include "Comparer.hh"
#include "celeritas_test.hh"
#include "testdetail/TestMacrosImpl.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

struct EventResult
{
    //// TYPES ////

    using Real3 = Array<double, 3>;

    //// DATA ////

    std::vector<Real3> pos;
    std::vector<double> time;
    std::vector<int> vtx;
    std::vector<int> pdg;
    std::vector<double> energy;
    std::vector<Real3> dir;
    std::vector<double> mass;

    //// REFERENCE RESULT ////

    //! Tolerances for comparison with defaults
    struct Tol
    {
        double energy{Test::fine_eps};
        double dir{Test::coarse_eps};
    };

    //! Implement \c EXPECT_REF_NEAR and mark as a reference result via traits
    static constexpr char class_str[] = "EventResult";

    //! Implement \c print_expected
    friend std::ostream& operator<<(std::ostream& os, EventResult const& ref)
    {
        return os << '{' << CELER_ATTR_REPR(ref, pos) << ", "
                  << CELER_ATTR_REPR(ref, time) << ", "
                  << CELER_ATTR_REPR(ref, vtx) << ", "
                  << CELER_ATTR_REPR(ref, pdg) << ", "
                  << CELER_ATTR_REPR(ref, energy) << ", "
                  << CELER_ATTR_REPR(ref, dir) << ", "
                  << CELER_ATTR_REPR(ref, mass) << '}';
    }

    //! Implement \c EXPECT_REF_NEAR
    friend bool check_ref_eq(AssertionHelper& helper,
                             EventResult const& expected,
                             EventResult const& actual,
                             Tol const& tol)
    {
        RefComparer<EventResult> compare{helper, expected, actual};
        CELER_INVOKE_ATTR(compare.is_soft_eq, EventResult, pos);
        CELER_INVOKE_ATTR(compare.is_soft_eq, EventResult, time);
        CELER_INVOKE_ATTR(compare.is_eq, EventResult, vtx);
        CELER_INVOKE_ATTR(compare.is_eq, EventResult, pdg);
        CELER_INVOKE_ATTR(compare.is_soft_eq, EventResult, energy, tol.energy);
        CELER_INVOKE_ATTR(compare.is_soft_eq, EventResult, dir, tol.dir);
        CELER_INVOKE_ATTR(compare.is_soft_eq, EventResult, mass);
        return static_cast<bool>(compare);
    }

    //! Implement \c EXPECT_REF_EQ
    friend bool check_ref_eq(AssertionHelper& helper,
                             EventResult const& expected,
                             EventResult const& actual)
    {
        return check_ref_eq(helper, expected, actual, {});
    }
};

class HepMC3PrimaryGeneratorTest : public SimpleCmsTestBase
{
  protected:
    std::vector<EventResult> read_all(G4VPrimaryGenerator& gen, int num_events)
    {
        using ClhepLength = lengthunits::ClhepLength;

        CELER_EXPECT(num_events > 0);
        std::vector<EventResult> result;
        for (auto ev_id : range(num_events))
        {
            G4Event event;
            event.SetEventID(ev_id);
            EXPECT_EQ(0, event.GetNumberOfPrimaryVertex());
            gen.GeneratePrimaryVertex(&event);
            EventResult& event_result = result.emplace_back();
            size_type num_primaries{0};
            for (auto vtx_id : range(event.GetNumberOfPrimaryVertex()))
            {
                G4PrimaryVertex* vtx = event.GetPrimaryVertex(vtx_id);
                CELER_ASSERT(vtx);
                auto pos_cm = to_cm(make_quantity_array<ClhepLength>(
                    to_array(vtx->GetPosition())));
                event_result.pos.push_back(pos_cm);
                event_result.time.push_back(
                    native_from_geant<units::ClhepTime>(vtx->GetT0())
                    / units::nanosecond);
                num_primaries += vtx->GetNumberOfParticle();
                for (auto j : range(vtx->GetNumberOfParticle()))
                {
                    G4PrimaryParticle* p = vtx->GetPrimary(j);
                    CELER_ASSERT(p);
                    event_result.vtx.push_back(vtx_id);
                    event_result.pdg.push_back(p->GetPDGcode());
                    event_result.energy.push_back(p->GetKineticEnergy());
                    event_result.mass.push_back(p->GetMass());

                    auto dir = to_array(p->GetMomentumDirection());
                    event_result.dir.push_back(dir);
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

TEST_F(HepMC3PrimaryGeneratorTest, no_vertex)
{
    HepMC3PrimaryGenerator generator(
        this->test_data_path("celeritas", "event-novtx.hepmc3"));
    EXPECT_EQ(3, generator.NumEvents());
    auto result = this->read_all(generator, generator.NumEvents());
    std::vector<EventResult> ref = {
        {/* pos = */ {{0, 0, 50}},
         /* time = */ {4.1028383709373},
         /* vtx = */ {0, 0, 0, 0, 0},
         /* pdg = */ {22, 22, 22, 22, 22},
         /* energy = */ {1000, 1000, 1000.0000107896, 999.99998921041, 1000},
         /* dir = */
         {{0.51986662883182, -0.42922054653912, -0.7385854118893},
          {0.73395459362461, 0.18726575230281, 0.65287226354916},
          {-0.40053358241289, -0.081839341451527, 0.91261994913013},
          {-0.51571621404849, 0.125780323886, 0.84747631040084},
          {-0.50829382297518, 0.51523183959, -0.69005328852051}},
         /* mass = */ {0, 0, 0, 0, 0}},
        {/* pos = */ {{0, 0, 50}},
         /* time = */ {4.1028383709373},
         /* vtx = */ {0, 0, 0, 0, 0},
         /* pdg = */ {22, 22, 22, 22, 22},
         /* energy = */
         {999.99998921041,
          1000.0000107896,
          999.99998921041,
          999.99998474121,
          1000},
         /* dir = */
         {{0.25183128938865, -0.20216120822227, -0.94642054477646},
          {-0.25247976713164, 0.94617275706344, -0.20251192799469},
          {0.34066344768752, -0.90517210955886, 0.25418864547108},
          {0.83192692739206, -0.5433000688087, 0.11279460409292},
          {0.23445050379268, -0.36984950141989, -0.89902408620171}},
         /* mass = */ {0, 0, 0, 0, 0}},
        {/* pos = */ {{0, 0, 50}},
         /* time = */ {4.1028383709373},
         /* vtx = */ {0, 0, 0, 0, 0},
         /* pdg = */ {22, 22, 22, 22, 22},
         /* energy = */ {1000, 999.99998921041, 1000, 1000.0000152588, 999.99998474121},
         /* dir = */
         {{0.17562103525404, -0.47618127524474, 0.86163138585047},
          {-0.60694965222664, 0.69697036165837, 0.38189584264792},
          {0.51336099422575, 0.54197742781709, 0.66537279576514},
          {-0.36655746358148, 0.80035990693978, 0.47440451647941},
          {-0.78969793730749, -0.54961247282688, -0.27258631206541}},
         /* mass = */ {0, 0, 0, 0, 0}}};
    EventResult::Tol tol;
    tol.energy = coarse_eps;
    tol.dir = coarse_eps;
    EXPECT_REF_NEAR(ref, result, tol);
}

TEST_F(HepMC3PrimaryGeneratorTest, multiple_vertex)
{
    HepMC3PrimaryGenerator generator(
        this->test_data_path("celeritas", "event-variety.hepmc3"));
    EXPECT_EQ(3, generator.NumEvents());
    auto result = this->read_all(generator, generator.NumEvents());
    // print_expected(result);

    std::vector<EventResult> ref
        = {{/* pos = */ {{0, 0, 0}, {0, 0, 0}},
            /* time = */ {0, 0},
            /* vtx = */ {0, 1, 1},
            /* pdg = */ {22, 1, -2},
            /* energy = */ {4151.3789242904, 29651.503768773, 56547.034479342},
            /* dir = */
            {{-0.90094709007965, 0.02669997932835, -0.43310674432625},
             {-0.082735048064663, 0.97508922087171, 0.20580554696494},
             {0.0702815376096, -0.87804026971226, -0.47339813078935}},
            /* mass = */ {0, -1, -1}},
           {/* pos = */ {{0, 0, 1}, {0, 0, 1}},
            /* time = */ {0, 0},
            /* vtx = */ {0, 1, 1},
            /* pdg = */ {22, 1, -2},
            /* energy = */ {4151.3789242904, 29651.503768773, 56547.034479342},
            /* dir = */
            {{-0.90094709007965, 0.02669997932835, -0.43310674432625},
             {-0.082735048064663, 0.97508922087171, 0.20580554696494},
             {0.0702815376096, -0.87804026971226, -0.47339813078935}},
            /* mass = */ {0, -1, -1}},
           {/* pos = */ {{1, 2, 3}, {1, 2, 3}},
            /* time = */ {0.13342563807926, 0.13342563807926},
            /* vtx = */ {0, 1, 1},
            /* pdg = */ {22, 1, -2},
            /* energy = */ {4151.3789242904, 29651.503768773, 56547.034479342},
            /* dir = */
            {{-0.90094709007965, 0.02669997932835, -0.43310674432625},
             {-0.082735048064663, 0.97508922087171, 0.20580554696494},
             {0.0702815376096, -0.87804026971226, -0.47339813078935}},
            /* mass = */ {0, -1, -1}}};
    EXPECT_REF_EQ(ref, result);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
