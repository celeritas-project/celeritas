//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/AlongStepTestBase.cc
//---------------------------------------------------------------------------//
#include "AlongStepTestBase.hh"

#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/LogContextException.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/PhysicsStepUtils.hh"
#include "celeritas/track/TrackInitData.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/track/TrackInitUtils.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
auto AlongStepTestBase::run(Input const& inp, size_type num_tracks) -> RunResult
{
    CELER_EXPECT(inp);
    CELER_EXPECT(num_tracks > 0);

    // Create states (single thread)
    CollectionStateStore<CoreStateData, MemSpace::host> states{
        this->core()->host_ref(), StreamId{0}, num_tracks};
    auto& core_params = this->core()->host_ref();
    auto& core_states = states.ref();
    CELER_ASSERT(core_states);

    {
        // Create primary from input (separate event IDs)
        Primary p;
        p.particle_id = inp.particle_id;
        p.energy = inp.energy;
        p.position = inp.position;
        p.direction = inp.direction;
        p.time = inp.time;

        std::vector<Primary> primaries(num_tracks, p);
        for (auto i : range(num_tracks))
        {
            primaries[i].event_id = EventId{i};
            primaries[i].track_id = TrackId{i};
        }

        // Primary -> track initializer -> track
        extend_from_primaries(*this->core(), core_states, make_span(primaries));
        initialize_tracks(*this->core(), core_states);
    }

    // Set remaining MFP and cached MSC range properties
    for (auto tid : range(ThreadId{num_tracks}))
    {
        CoreTrackView track{core_params, core_states, tid};
        auto phys = track.make_physics_view();
        phys.interaction_mfp(inp.phys_mfp);
        if (inp.msc_range)
        {
            phys.msc_range(inp.msc_range);
        }
    }

    auto const& am = *this->action_reg();
    {
        // Call pre-step action to set range, physics step
        auto prestep_action_id = am.find_action("pre-step");
        CELER_ASSERT(prestep_action_id);
        auto const& prestep_action
            = dynamic_cast<ExplicitActionInterface const&>(
                *am.action(prestep_action_id));
        CELER_TRY_HANDLE(prestep_action.execute(*this->core(), core_states),
                         LogContextException{this->output_reg().get()});

        // Call along-step action
        auto const& along_step = *this->along_step();
        CELER_TRY_HANDLE(along_step.execute(*this->core(), core_states),
                         LogContextException{this->output_reg().get()});
    }

    // Process output
    RunResult result;
    std::map<ActionId, int> actions;
    for (auto tid : range(ThreadId{num_tracks}))
    {
        CoreTrackView track{core_params, core_states, tid};
        auto sim = track.make_sim_view();
        auto particle = track.make_particle_view();
        auto geo = track.make_geo_view();
        auto phys = track.make_physics_view();

        result.eloss += value_as<MevEnergy>(inp.energy)
                        - value_as<MevEnergy>(particle.energy());
        result.displacement += distance(geo.pos(), inp.position);
        result.angle += dot_product(geo.dir(), inp.direction);
        result.time += sim.time();
        result.step += sim.step_limit().step;
        result.mfp += inp.phys_mfp - phys.interaction_mfp();
        result.alive += sim.status() == TrackStatus::alive ? 1 : 0;
        actions[sim.step_limit().action] += 1;
    }

    real_type norm = 1 / real_type(num_tracks);
    result.eloss *= norm;
    result.displacement *= norm;
    result.angle *= norm;
    result.time *= norm;
    result.step *= norm;
    result.mfp *= norm;
    result.alive *= norm;

    if (actions.size() == 1)
    {
        result.action = am.id_to_label(actions.begin()->first);
    }
    else
    {
        // Stochastic action from along-step!
        std::ostringstream os;
        os << '{'
           << join_stream(actions.begin(),
                          actions.end(),
                          ", ",
                          [&am, norm](std::ostream& os, auto const& kv) {
                              os << '"' << am.id_to_label(kv.first)
                                 << "\": " << kv.second * norm;
                          })
           << '}';
    }

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
            "EXPECT_SOFT_EQ("
         << repr(this->mfp) << ", result.mfp);\n"
            "EXPECT_SOFT_EQ("
         << repr(this->alive)
         << ", result.alive);\n";
    if (!this->action.empty() && this->action.front() == '{')
        cout << "// ";
    cout << "EXPECT_EQ(" << repr(this->action)
         << ", result.action);\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
