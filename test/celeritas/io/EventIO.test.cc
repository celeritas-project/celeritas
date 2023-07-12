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

using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class EventIO : public Test, public ::testing::WithParamInterface<char const*>
{
  protected:
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

    std::shared_ptr<ParticleParams> particles_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_P(EventIO, read_all_formats)
{
    std::string const ext = this->GetParam();
    std::string filename
        = this->test_data_path("celeritas", "event-record." + ext);

    // Determine the event record format and open the file
    EventReader read_event(filename, particles_);

    int const expected_pdg[] = {2212, 1, 2212, -2, 22, -24, 1, -2};

    double const expected_energy[] = {
        7.e6, 3.2238e4, 7.e6, 5.7920e4, 4.233e3, 8.5925e4, 2.9552e4, 5.6373e4};

    double const expected_direction[][3] = {
        {0, 0, 1},
        {2.326451417389850e-2, -4.866936365179566e-2, 9.985439676959555e-1},
        {0, 0, -1},
        {-5.260794237813896e-2, -3.280442747570201e-1, -9.431963518790131e-1},
        {-9.009470900796461e-1, 2.669997932835038e-2, -4.331067443262500e-1},
        {5.189457940206315e-2, -7.074356638330033e-1, -7.048700122475354e-1},
        {-8.273504806466310e-2, 9.750892208717103e-1, 2.058055469649411e-1},
        {7.028153760960004e-2, -8.780402697122620e-1, -4.733981307893478e-1}};

    // Read events from the event record
    int event_count = 0;
    std::vector<Primary> primaries;
    do
    {
        primaries = read_event();
        ASSERT_TRUE(primaries.empty()
                    || primaries.size() == std::size(expected_pdg));
        for (auto i : range(primaries.size()))
        {
            auto const& primary = primaries[i];

            // Check that the particle types were read correctly
            EXPECT_EQ(
                expected_pdg[i],
                particles_->id_to_pdg(primary.particle_id).unchecked_get());

            // Check that the event IDs match
            EXPECT_EQ(event_count, primary.event_id.unchecked_get());

            // Check that the position, direction, and energy were read
            // correctly
            double const expected_position[] = {0, 0, 0};
            EXPECT_VEC_SOFT_EQ(expected_position, primary.position);
            EXPECT_VEC_SOFT_EQ(expected_direction[i], primary.direction);
            EXPECT_DOUBLE_EQ(expected_energy[i],
                             value_as<MevEnergy>(primary.energy));
            EXPECT_EQ(0, primary.time);
        }
        ++event_count;
    } while (!primaries.empty());
    // Event reader should keep returning an empty vector
    primaries = read_event();
    EXPECT_TRUE(primaries.empty());
}

TEST_P(EventIO, write_read)
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
            primaries[i].track_id = TrackId{i * 2};
        }
        return primaries;
    }();

    std::string filename
        = this->make_unique_filename(std::string{"."} + this->GetParam());
    cout << "filename: " << filename << endl;

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
            ScopedLogStorer scoped_log_{&celeritas::world_logger()};
            write_event(primaries);
            static char const* const expected_log_messages[]
                = {"Overwriting primary event IDs with 2: 1"};
            EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
            static char const* const expected_log_levels[] = {"warning"};
            EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
        }
    }

    // Read events
    std::vector<int> pdg;
    std::vector<double> energy;
    std::vector<double> pos;
    std::vector<double> dir;
    std::vector<double> time;
    std::vector<int> event;
    std::vector<int> track;
    {
        EventReader read_event(filename, particles_);
        do
        {
            primaries = read_event();
            for (auto const& p : primaries)
            {
                pdg.push_back(
                    particles_->id_to_pdg(p.particle_id).unchecked_get());
                energy.push_back(p.energy.value());
                pos.insert(pos.end(), p.position.begin(), p.position.end());
                dir.insert(dir.end(), p.direction.begin(), p.direction.end());
                time.push_back(p.time);
                event.push_back(p.event_id.unchecked_get());
                track.push_back(p.track_id.unchecked_get());
            }
        } while (!primaries.empty());
    }

#if 0
    PRINT_EXPECTED(pdg);
    PRINT_EXPECTED(energy);
    PRINT_EXPECTED(pos);
    PRINT_EXPECTED(dir);
    PRINT_EXPECTED(time);
    PRINT_EXPECTED(event);
    PRINT_EXPECTED(track);
#endif
    // clang-format off
    static int const expected_pdg[] = {22, 2212, 22, 2212, 22, 2212, 22, 2212, 2212, 22};
    static double const expected_energy[] = {1.23, 2.34, 1.23, 2.34, 1.23, 2.34, 1.23, 2.34, 3.45, 1.23};
    static double const expected_pos[] = {2, 4, 5, -3, -4, 5, 2, 4, 5, 2, 4, 5, 2, 4, 5, -3, -4, 5, 2, 4, 5, 2, 4, 5, 2, 4, 5, 2, 4, 5};
    static double const expected_dir[] = {1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0};
    static double const expected_time[] = {5.67e-09, 5.78e-09, 5.67e-09, 5.67e-09, 5.67e-09, 5.78e-09, 5.67e-09, 5.67e-09, 5.67e-09, 5.67e-09};
    static int const expected_event[] = {0, 0, 0, 0, 1, 1, 1, 1, 1, 2};
    static int const expected_track[] = {0, 1, 2, 3, 0, 1, 2, 3, 4, 0};
    // clang-format on

    EXPECT_VEC_EQ(expected_pdg, pdg);
    EXPECT_VEC_SOFT_EQ(expected_energy, energy);
    EXPECT_VEC_SOFT_EQ(expected_pos, pos);
    EXPECT_VEC_SOFT_EQ(expected_dir, dir);
    EXPECT_VEC_SOFT_EQ(expected_time, time);
    EXPECT_VEC_EQ(expected_event, event);
    EXPECT_VEC_EQ(expected_track, track);
}

INSTANTIATE_TEST_SUITE_P(EventReaderTests,
                         EventIO,
                         testing::Values("hepmc3", "hepmc2", "hepevt"));

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
