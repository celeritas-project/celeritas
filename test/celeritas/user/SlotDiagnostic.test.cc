//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SlotDiagnostic.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/user/SlotDiagnostic.hh"

#include <fstream>
#include <nlohmann/json.hpp>

#include "corecel/io/Repr.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "SimpleLoopTestBase.hh"
#include "celeritas_test.hh"
#include "../TestEm3Base.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
char pid_to_char(int i)
{
    switch (i)
    {
        case -2:  // errored
            return 'x';
        case -1:  // empty
            return ' ';
        case 0:  // gamma
            return '0';
        case 1:  // electron
            return '-';
        case 2:  // positron
            return '+';
        default:
            return '!';
    }
}

//---------------------------------------------------------------------------//

class SlotDiagnosticTest : virtual public SimpleLoopTestBase
{
  protected:
    struct RunResult
    {
        std::vector<std::string> labels;
        std::vector<std::string> slots;
        void print_expected() const;
    };

    void SetUp() override
    {
        basename_ = this->make_unique_filename("-");
        slot_diagnostic_
            = SlotDiagnostic::make_and_insert(*this->core(), basename_);
    }

    template<MemSpace M>
    RunResult run(size_type num_tracks, size_type num_steps)
    {
        this->run_impl<M>(num_tracks, num_steps);
        return this->parse_output();
    }

    RunResult parse_output() const;

  private:
    std::shared_ptr<SlotDiagnostic> slot_diagnostic_;
    std::string basename_;
};

auto SlotDiagnosticTest::parse_output() const -> RunResult
{
    using nlohmann::json;

    RunResult out;

    out.labels = [this] {
        // Read metadata
        std::ifstream infile(basename_ + "metadata.json");
        CELER_VALIDATE(infile, << "failed to load metadata");
        auto md = json::parse(infile);
        std::vector<std::string> labels;
        md.at("label").get_to(labels);
        return labels;
    }();

    out.slots = [this] {
        // Read data from stream 0
        std::vector<std::string> result;
        std::ifstream infile(basename_ + "0.jsonl");
        CELER_VALIDATE(infile, << "failed to load result");
        std::string line;
        while (std::getline(infile, line))
        {
            auto vals = json::parse(line);
            std::string converted;
            for (auto&& v : vals)
            {
                converted.push_back(pid_to_char(v.get<int>()));
            }
            result.emplace_back(std::move(converted));
        }
        return result;
    }();
    return out;
}

void SlotDiagnosticTest::RunResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static char const* const expected_labels[] = "
         << repr(this->labels)
         << ";\n"
            "EXPECT_VEC_EQ(expected_labels, "
            "result.labels);\n"
            "static char const* const expected_slots[] = "
         << repr(this->slots)
         << ";\n"
            "EXPECT_VEC_EQ(expected_slots, "
            "result.slots);\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
class TestEm3SlotTest : virtual public TestEm3Base,
                        virtual public SlotDiagnosticTest
{
    double initial_occupancy() const override { return 0.125; }

    auto build_init() -> SPConstTrackInit override
    {
        TrackInitParams::Input input;
        input.capacity = 32768;
        input.max_events = 16;
        input.track_order = TrackOrder::partition_charge;
        return std::make_shared<TrackInitParams>(input);
    }

    VecPrimary make_primaries(size_type count) const override
    {
        Primary p;
        p.energy = units::MevEnergy{50};
        p.event_id = EventId{0};
        p.position = from_cm({-22, 0, 0});
        p.direction = {1, 0, 0};
        p.time = 0;

        Array<ParticleId, 3> const particles = {
            this->particle()->find(pdg::electron()),
            this->particle()->find(pdg::positron()),
            this->particle()->find(pdg::gamma()),
        };
        CELER_ASSERT(particles[0] && particles[1] && particles[2]);

        std::vector<Primary> result(count, p);
        for (auto i : range(count))
        {
            result[i].track_id = TrackId{i};
            result[i].particle_id = particles[i % particles.size()];
        }
        return result;
    }
};

//---------------------------------------------------------------------------//

TEST_F(TestEm3SlotTest, host)
{
    auto result = this->run<MemSpace::host>(32, 64);

    static char const* const expected_labels[] = {"gamma", "e-", "e+"};
    EXPECT_VEC_EQ(expected_labels, result.labels);
    static char const* const expected_slots[] = {
        "0                            -+-", "0                            -+-",
        "0000                         -+-", "0000000                      -+-",
        "00000000                  -+--+-", "0000000000                -+--+-",
        "00000000000            ----+--+-", "00000000000000           --+--+-",
        "0000000000000000      -+---+--+-", "0000000000000000000----+---+--+-",
        "0000000000 0000 000----+---+--+-", "0000000000 0000 000----+---+--+-",
        "0000000000000000-00----+---+--+-", "0000000000000000000-00-+---+--+-",
        "0000000000000000000-00-+---+--+-", "0000000000000000000-0000-0-+--+-",
        "0000000000000000000-0000-0-+--+-", "0000000000000000000-0000-0-+--+-",
        "0000000000000000000-0000-0-+--+-", "0000000000000000000-0000-0-+--+-",
        "000000000000000-000-0000-0-+--+-", "000000000000000--00-00-0-0-+--+-",
        "000000000000000--00-0--0-0++--+-", "0000000000000000-00-0--0-0++--+-",
        "0000000000000000-0000--0---+--+-", "0000000000000000-0000--0---+--+-",
        "00000000000000-0-0000--0---+--+-", "00000000000000-0-00-0--0---+--+-",
        "-00-0000-00000-0-00-0--0---+--+-", "000-0000-00000-0-00-0--0---+--+-",
        "00000000-0-000-0--0-0--0---+--+-", "0000000000-000----0-0--0---+--+-",
        "0000000000--00----0-0--0---+--+-", "00000000000000----0-0--0---+--+-",
        "--0-0000000000----0-0--0---+--+-", "00--0000000000-+--0-0--0---+--+-",
        "00--0000000000-+--0-0--0---+--+-", "00-00-00000000-+--0-0--0---+--+-",
        "00-00-00000-00-+--0-0--0---+--+-", "00-00000000000-+--0-0  0- -+-- -",
        "00-00000000 0 -+ -0----0---+--+-", "00-0000000  0 -+--0----0---+--+-",
        "00-0000000000 -  - -  -0---+--+-", "00-00000000 0 -  - -   0------+-",
        "00-00000000000-  - -   0--  - --", "00-00000000000-00- -   0-- -----",
        "00-000 0000000- 0- - --0--------", "00-00000000000-00- - --0--------",
        "00 00000 00 00-00- -  -0--------", "00 00 00 0  0 -00- -   0- ------",
        "00 00 00 0  0 - 0- -   0- ------", "00000 00 0  0 - 0- -   0- ------",
        "00  0 00    0 - 0- - --0------+-", "00  0  0    0 - 0- -   0 -----+-",
        "00000  0    0 - 0- -   0  -- ---", "000000 0    0 - 0- -   0 -------",
        "000000 0    0 - 0  -   0  --  --", " 0000  0    0 -    -   0  ------",
        " 00    0    0 -    -   0   - ---", " 00    0    0 -        0   -   -",
        " 00    0    0 -            -  --", " 00    0    0 -            -    ",
        " 0     0    0 -            -  --", "       0    0 -            -   -",
    };
    EXPECT_VEC_EQ(expected_slots, result.slots);
}

TEST_F(TestEm3SlotTest, TEST_IF_CELER_DEVICE(device))
{
    auto result = this->run<MemSpace::host>(64, 16);

    static char const* const expected_labels[] = {"gamma", "e-", "e+"};
    EXPECT_VEC_EQ(expected_labels, result.labels);
    static char const* const expected_slots[] = {
        "00                                                        -+-+-+",
        "00                                                        -+-+-+",
        "00000000                                                  -+-+-+",
        "000000000000                                             --+-+-+",
        "000000000000000                                       -+-+-+-+-+",
        "00000000000000000000                            ---+---+-+-+-+-+",
        "00000000000000000000000                         ---+---+-+-+-+-+",
        "000000000000000000000000000                --+-----+---+-+-+-+-+",
        "0000000000000000000000000000000             -+-----+---+-+-+-+-+",
        "000000000000000000000000000000000       -----++----+---+-+-+-+-+",
        "00000000000000000000000000000000000   --+----++----+---+-+-+-+-+",
        "000000000000000000000000000000000000000-+----++----+---+-+-+-+-+",
        "0000000000000000000000000000000000-0000-+----++----+---+-+-+-+-+",
        "0000000000000000000000000-00000-00-0000-+----++--------+-+-+-+-+",
        "0000000000000000000000000-0000000000000-+----++--------+-+-+-+-+",
        "0000000000000000000000000-000000-00-000------++--------+-+-+-+-+",
    };
    EXPECT_VEC_EQ(expected_slots, result.slots);
}

TEST_F(TestEm3SlotTest, DISABLED_long_demo)
{
    if (celeritas::device())
    {
        this->run<MemSpace::device>(2046, 256);
    }
    else
    {
        this->run<MemSpace::host>(2046, 256);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
