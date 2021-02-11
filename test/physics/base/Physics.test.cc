//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Physics.test.cc
//---------------------------------------------------------------------------//
#include "physics/base/PhysicsParams.hh"
#include "physics/base/PhysicsTrackView.hh"

#include "celeritas_test.hh"
#include "base/Range.hh"
#include "base/PieStateStore.hh"
#include "physics/base/ParticleParams.hh"
#include "PhysicsTestBase.hh"

using namespace celeritas;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// PHYSICS BASE CLASS
//---------------------------------------------------------------------------//

class PhysicsParamsTest : public PhysicsTestBase
{
};

TEST_F(PhysicsParamsTest, accessors)
{
    const PhysicsParams& p = *this->physics();

    EXPECT_EQ(5, p.num_processes());
    EXPECT_EQ(2 + 1 + 3 + 2 + 2, p.num_models());
    EXPECT_EQ(3, p.max_particle_processes());

    // Test process names after construction
    std::vector<std::string> process_names;
    for (auto process_idx : range(p.num_processes()))
    {
        process_names.push_back(p.process(ProcessId{process_idx}).label());
    }
    const std::string expected_process_names[]
        = {"scattering", "absorption", "purrs", "hisses", "meows"};
    EXPECT_VEC_EQ(expected_process_names, process_names);

    // Test model names after construction
    std::vector<std::string> model_names;
    for (auto model_idx : range(p.num_models()))
    {
        model_names.push_back(p.model(ModelId{model_idx}).label());
    }
    const std::string expected_model_names[]
        = {"MockModel(0, p=0, emin=1e-06, emax=100)",
           "MockModel(1, p=1, emin=1, emax=100)",
           "MockModel(2, p=0, emin=1e-06, emax=100)",
           "MockModel(3, p=1, emin=0.001, emax=1)",
           "MockModel(4, p=1, emin=1, emax=10)",
           "MockModel(5, p=1, emin=10, emax=100)",
           "MockModel(6, p=2, emin=0.001, emax=1)",
           "MockModel(7, p=2, emin=1, emax=100)",
           "MockModel(8, p=1, emin=0.001, emax=10)",
           "MockModel(9, p=2, emin=0.001, emax=10)"};
    EXPECT_VEC_EQ(expected_model_names, model_names);

    // Test host-accessible process map
    std::vector<std::string> process_map;
    for (auto particle_idx : range(this->particles()->size()))
    {
        std::string prefix
            = this->particles()->id_to_label(ParticleId{particle_idx});
        prefix.push_back(':');
        for (ProcessId process_id : p.processes(ParticleId{particle_idx}))
        {
            process_map.push_back(prefix + process_names[process_id.get()]);
        }
    }
    const std::string expected_process_map[] = {"gamma:scattering",
                                                "gamma:absorption",
                                                "celeriton:scattering",
                                                "celeriton:purrs",
                                                "celeriton:meows",
                                                "anti-celeriton:hisses",
                                                "anti-celeriton:meows"};
    EXPECT_VEC_EQ(expected_process_map, process_map);
}

//---------------------------------------------------------------------------//
// PHYSICS TRACK VIEW (HOST)
//---------------------------------------------------------------------------//

class PhysicsTrackViewHostTest : public PhysicsTestBase
{
    using Base = PhysicsTestBase;

  public:
    //!@{
    //! Type aliases
    using StateStore = PieStateStore<PhysicsStateData, MemSpace::host>;
    using ParamsHostRef
        = PhysicsParamsData<Ownership::const_reference, MemSpace::host>;
    using MevEnergy = celeritas::units::MevEnergy;
    //!@}

    void SetUp() override
    {
        Base::SetUp();

        // Make one state per particle
        auto state_size = this->particles()->size();

        CELER_ASSERT(this->physics());
        params_ref = this->physics()->host_pointers();
        state      = StateStore(*this->physics(), state_size);

        // Save mapping of process label -> ID
        for (auto process_idx : range(this->physics()->num_processes()))
        {
            ProcessId id{process_idx};
            process_names[this->physics()->process(id).label()] = id;
        }
    }

    PhysicsTrackView make_track_view(const char* particle, MaterialId mid)
    {
        CELER_EXPECT(particle && mid);

        auto pid = this->particles()->find(particle);
        CELER_ASSERT(pid);
        CELER_ASSERT(pid.get() < state.size());

        ThreadId tid((pid.get() + 1) % state.size());

        // Construct (thread depends on particle here to shake things up) and
        // initialize
        PhysicsTrackView phys(params_ref, state.ref(), pid, mid, tid);
        phys = PhysicsTrackInitializer{};

        return phys;
    }

    ParticleProcessId
    find_ppid(const PhysicsTrackView& track, const char* label) const
    {
        auto iter = process_names.find(label);
        CELER_VALIDATE(iter != process_names.end(),
                       "No process named " << label);
        ProcessId pid = iter->second;
        for (auto pp_idx : range(track.num_particle_processes()))
        {
            if (track.process(ParticleProcessId{pp_idx}) == pid)
                return ParticleProcessId{pp_idx};
        }
        return {};
    }

    ParamsHostRef                    params_ref;
    StateStore                       state;
    std::map<std::string, ProcessId> process_names;
};

TEST_F(PhysicsTrackViewHostTest, accessors)
{
    PhysicsTrackView gamma = this->make_track_view("gamma", MaterialId{0});
    PhysicsTrackView celer = this->make_track_view("celeriton", MaterialId{1});
    const PhysicsTrackView& gamma_cref = gamma;

    // Interaction MFP
    {
        EXPECT_FALSE(gamma_cref.has_interaction_mfp());

        gamma.interaction_mfp(1.234);
        celer.interaction_mfp(2.345);
        EXPECT_DOUBLE_EQ(1.234, gamma_cref.interaction_mfp());
        EXPECT_DOUBLE_EQ(2.345, celer.interaction_mfp());
    }

    // Cross sections
    {
        const PhysicsTrackView& gamma_cref = gamma;

        gamma.per_process_xs(ParticleProcessId{0}) = 1.2;
        gamma.per_process_xs(ParticleProcessId{1}) = 10.0;
        celer.per_process_xs(ParticleProcessId{0}) = 100.0;
        EXPECT_DOUBLE_EQ(1.2, gamma_cref.per_process_xs(ParticleProcessId{0}));
        EXPECT_DOUBLE_EQ(10.0, gamma_cref.per_process_xs(ParticleProcessId{1}));
        EXPECT_DOUBLE_EQ(100.0, celer.per_process_xs(ParticleProcessId{0}));
    }
}

TEST_F(PhysicsTrackViewHostTest, processes)
{
    // Gamma
    {
        const PhysicsTrackView phys
            = this->make_track_view("gamma", MaterialId{0});

        EXPECT_EQ(2, phys.num_particle_processes());
        const ParticleProcessId scat_ppid{0};
        const ParticleProcessId abs_ppid{1};
        EXPECT_EQ(ProcessId{0}, phys.process(scat_ppid));
        EXPECT_EQ(ProcessId{1}, phys.process(abs_ppid));
    }

    // Celeriton
    {
        const PhysicsTrackView phys
            = this->make_track_view("celeriton", MaterialId{0});

        EXPECT_EQ(3, phys.num_particle_processes());
        const ParticleProcessId scat_ppid{0};
        const ParticleProcessId purr_ppid{1};
        const ParticleProcessId meow_ppid{2};
        EXPECT_EQ(ProcessId{0}, phys.process(scat_ppid));
        EXPECT_EQ(ProcessId{2}, phys.process(purr_ppid));
        EXPECT_EQ(ProcessId{4}, phys.process(meow_ppid));
    }

    // Anti-celeriton
    {
        const PhysicsTrackView phys
            = this->make_track_view("anti-celeriton", MaterialId{1});

        EXPECT_EQ(2, phys.num_particle_processes());
        const ParticleProcessId hiss_ppid{0};
        const ParticleProcessId meow_ppid{1};
        EXPECT_EQ(ProcessId{3}, phys.process(hiss_ppid));
        EXPECT_EQ(ProcessId{4}, phys.process(meow_ppid));
    }
}

TEST_F(PhysicsTrackViewHostTest, value_grids)
{
    std::vector<int> grid_ids;

    for (const char* particle : {"gamma", "celeriton", "anti-celeriton"})
    {
        for (auto mat_idx : range(this->materials()->size()))
        {
            const PhysicsTrackView phys
                = this->make_track_view(particle, MaterialId{mat_idx});

            for (auto pp_idx : range(phys.num_particle_processes()))
            {
                for (ValueGridType vgt : range(ValueGridType::size_))
                {
                    auto id = phys.value_grid(vgt, ParticleProcessId{pp_idx});
                    grid_ids.push_back(id ? id.get() : -1);
                }
            }
        }
    }

    // Grid IDs should be unique if they exist. Gammas should have fewer
    // because there aren't any slowing down/range limiters.
    const int expected_grid_ids[]
        = {0,  -1, -1, 3,  -1, -1, 1,  -1, -1, 4,  -1, -1, 2,  -1, -1, 5,
           -1, -1, 6,  -1, -1, 9,  10, 11, 18, 19, 20, 7,  -1, -1, 12, 13,
           14, 21, 22, 23, 8,  -1, -1, 15, 16, 17, 24, 25, 26, 27, 28, 29,
           36, 37, 38, 30, 31, 32, 39, 40, 41, 33, 34, 35, 42, 43, 44};
    EXPECT_VEC_EQ(expected_grid_ids, grid_ids);
}

TEST_F(PhysicsTrackViewHostTest, calc_xs)
{
    // Cross sections: same across particle types, constant in energy, scale
    // according to material number density
    std::vector<real_type> xs;
    for (const char* particle : {"gamma", "celeriton"})
    {
        for (auto mat_idx : range(this->materials()->size()))
        {
            const PhysicsTrackView phys
                = this->make_track_view(particle, MaterialId{mat_idx});
            auto scat_ppid = this->find_ppid(phys, "scattering");
            auto id = phys.value_grid(ValueGridType::macro_xs, scat_ppid);
            ASSERT_TRUE(id);
            auto calc_xs = phys.make_calculator(id);
            xs.push_back(calc_xs(MevEnergy{1.0}));
        }
    }

    const double expected_xs[] = {0.0001, 0.001, 0.1, 0.0001, 0.001, 0.1};
    EXPECT_VEC_SOFT_EQ(expected_xs, xs);
}

TEST_F(PhysicsTrackViewHostTest, calc_range)
{
    // Default range and scaling
    EXPECT_SOFT_EQ(0.1 * units::centimeter, params_ref.scaling_min_range);
    EXPECT_SOFT_EQ(0.2, params_ref.scaling_fraction);
    std::vector<real_type> range;
    std::vector<real_type> step;

    // Range: increases with energy, constant with material
    for (const char* particle : {"celeriton", "anti-celeriton"})
    {
        const PhysicsTrackView phys
            = this->make_track_view(particle, MaterialId{0});
        auto meow_ppid = this->find_ppid(phys, "meows");
        ASSERT_TRUE(meow_ppid);
        auto id = phys.value_grid(ValueGridType::range, meow_ppid);
        ASSERT_TRUE(id);
        auto calc_range = phys.make_calculator(id);
        for (real_type energy : {0.01, 1.0, 1e2})
        {
            range.push_back(calc_range(MevEnergy{energy}));
            step.push_back(phys.range_to_step(range.back()));
        }
    }

    const double expected_range[] = {0.04, 4, 40, 0.04, 4, 40};
    const double expected_step[]  = {0.04, 0.958, 8.1598, 0.04, 0.958, 8.1598};
    EXPECT_VEC_SOFT_EQ(expected_range, range);
    EXPECT_VEC_SOFT_EQ(expected_step, step);
}

TEST_F(PhysicsTrackViewHostTest, model_finder)
{
    const PhysicsTrackView phys
        = this->make_track_view("celeriton", MaterialId{0});
    auto purr_ppid = this->find_ppid(phys, "purrs");
    ASSERT_TRUE(purr_ppid);
    auto find_model = phys.make_model_finder(purr_ppid);

    // See expected_model_names above
    EXPECT_FALSE(find_model(MevEnergy{0.999e-3}));
    EXPECT_EQ(3, find_model(MevEnergy{0.5}).unchecked_get());
    EXPECT_EQ(4, find_model(MevEnergy{5}).unchecked_get());
    EXPECT_EQ(5, find_model(MevEnergy{50}).unchecked_get());
    EXPECT_FALSE(find_model(MevEnergy{100.1}));
}
