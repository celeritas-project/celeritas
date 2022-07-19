//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/AlongStepTestBase.cc
//---------------------------------------------------------------------------//
#include "AlongStepTestBase.hh"

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/global/ActionManager.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/PhysicsStepUtils.hh"
#include "celeritas/track/TrackInitData.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/track/TrackInitUtils.hh"

using namespace celeritas;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
auto AlongStepTestBase::run(const Input& inp, size_type num_tracks) -> RunResult
{
    CELER_EXPECT(inp);
    CELER_EXPECT(num_tracks > 0);

    // Create states
    CollectionStateStore<CoreStateData, MemSpace::host> states{*this->core(),
                                                               num_tracks};
    CoreRef<MemSpace::host>                             core_ref;
    core_ref.params = this->core()->host_ref();
    core_ref.states = states.ref();
    CELER_ASSERT(core_ref);

    // Create primaries
    {
        Primary p;
        p.particle_id = inp.particle_id;
        p.energy      = inp.energy;
        p.position    = inp.position;
        p.direction   = inp.direction;
        p.time        = inp.time;
        p.track_id    = TrackId{0};

        // Create track initializers and add primaries
        TrackInitParams::Input inp;
        inp.primaries.assign(num_tracks, p);
        inp.capacity = num_tracks;
        for (auto i : range(num_tracks))
        {
            inp.primaries[i].event_id = EventId{i};
            inp.primaries[i].track_id = TrackId{i};
        }
        TrackInitParams init_params{std::move(inp)};

        TrackInitStateData<Ownership::value, MemSpace::host> init_states;
        resize(&init_states, init_params.host_ref(), num_tracks);

        celeritas::extend_from_primaries(init_params.host_ref(), &init_states);
        celeritas::initialize_tracks(core_ref, &init_states);
    }

    // Set remaining MFP
    for (auto tid : range(ThreadId{num_tracks}))
    {
        CoreTrackView track{core_ref.params, core_ref.states, tid};
        auto          phys = track.make_physics_view();
        phys.interaction_mfp(inp.phys_mfp);
    }

    {
        const auto& am = *this->action_mgr();

        // Call pre-step action to set range, physics step
        auto prestep_action = am.find_action("pre-step");
        CELER_ASSERT(prestep_action);
        am.invoke(prestep_action, core_ref);

        // Call along-step action
        const auto& along_step = *this->along_step();
        along_step.execute(core_ref);
    }

    // Process output
    RunResult result;
    for (auto tid : range(ThreadId{num_tracks}))
    {
        CoreTrackView track{core_ref.params, core_ref.states, tid};
        auto          sim      = track.make_sim_view();
        auto          particle = track.make_particle_view();
        auto          geo      = track.make_geo_view();

        result.eloss += value_as<MevEnergy>(inp.energy)
                        - value_as<MevEnergy>(particle.energy());
        result.displacement += celeritas::distance(geo.pos(), inp.position);
        result.angle += celeritas::dot_product(geo.dir(), inp.direction);
        result.time += sim.time();
        result.step += sim.step_limit().step;
    }

    real_type norm = 1 / real_type(num_tracks);
    result.eloss *= norm;
    result.displacement *= norm;
    result.angle *= norm;
    result.time *= norm;
    result.step *= norm;
    return result;
}

//---------------------------------------------------------------------------//
void AlongStepTestBase::RunResult::print_expected() const
{
    using std::cout;
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n";

    cout << "EXPECT_SOFT_EQ(" << repr(this->eloss)
         << ", result.eloss);\n"
            "EXPECT_SOFT_EQ("
         << repr(this->displacement)
         << ", result.displacement);\n"
            "EXPECT_SOFT_EQ("
         << repr(this->angle)
         << ", result.angle);\n"
            "EXPECT_SOFT_EQ("
         << repr(this->time)
         << ", result.time);\n"
            "EXPECT_SOFT_EQ("
         << repr(this->step)
         << ", result.step);\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
