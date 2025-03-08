//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstepTestBase.cc
//---------------------------------------------------------------------------//
#include "AlongStepTestBase.hh"

#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/LogContextException.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/phys/PhysicsStepUtils.hh"
#include "celeritas/track/ExtendFromPrimariesAction.hh"
#include "celeritas/track/TrackInitData.hh"
#include "celeritas/track/TrackInitParams.hh"

namespace celeritas
{
namespace test
{
namespace
{
using TimeSecond = RealQuantity<celeritas::units::Second>;
}

//---------------------------------------------------------------------------//
auto AlongStepTestBase::run(Input const& inp, size_type num_tracks) -> RunResult
{
    CELER_EXPECT(inp);
    CELER_EXPECT(num_tracks > 0);

    // Create states (single thread)
    CoreState<MemSpace::host> state{*this->core(), StreamId{0}, num_tracks};
    auto& core_params = this->core()->host_ref();
    auto& core_states = state.ref();
    CELER_ASSERT(core_states);

    {
        // Create primary from input (separate event IDs)
        Primary p;
        p.particle_id = inp.particle_id;
        p.energy = inp.energy;
        p.position = from_cm(inp.position);
        p.direction = inp.direction;
        p.time = native_value_from(TimeSecond{inp.time});

        std::vector<Primary> primaries(num_tracks, p);
        for (auto i : range(num_tracks))
        {
            primaries[i].event_id = EventId{i};
        }

        // Construct track initializers
        this->extend_from_primaries(make_span(primaries), &state);
    }

    // Construct tracks
    this->execute_action("initialize-tracks", &state);

    // Set remaining MFP and cached MSC range properties
    for (auto tid : range(ThreadId{num_tracks}))
    {
        CoreTrackView track{core_params, core_states, tid};
        auto phys = track.physics();
        phys.interaction_mfp(inp.phys_mfp);
        if (inp.msc_range)
        {
            phys.msc_range(inp.msc_range);
        }
    }

    // Call pre-step action to set range, physics step
    this->execute_action("pre-step", &state);

    // Call along-step action
    auto const& along_step = *this->along_step();
    CELER_TRY_HANDLE(along_step.step(*this->core(), state),
                     LogContextException{this->output_reg().get()});

    // Process output
    RunResult result;
    std::map<ActionId, int> actions;
    for (auto tid : range(ThreadId{num_tracks}))
    {
        CoreTrackView track{core_params, core_states, tid};
        auto sim = track.sim();
        auto particle = track.particle();
        auto geo = track.geometry();
        auto phys = track.physics();

        result.eloss += value_as<MevEnergy>(inp.energy)
                        - value_as<MevEnergy>(particle.energy());
        result.displacement += distance(to_cm(geo.pos()), inp.position);
        result.angle += dot_product(geo.dir(), inp.direction);
        result.time += native_value_to<TimeSecond>(sim.time()).value();
        result.step += to_cm(sim.step_length());
        result.mfp += inp.phys_mfp - phys.interaction_mfp();
        result.alive += sim.status() == TrackStatus::alive ? 1 : 0;
        actions[sim.post_step_action()] += 1;
    }

    real_type norm = 1 / real_type(num_tracks);
    result.eloss *= norm;
    result.displacement *= norm;
    result.angle *= norm;
    result.time *= norm;
    result.step *= norm;
    result.mfp *= norm;
    result.alive *= norm;

    auto action_id_label = [&areg = *this->action_reg()](ActionId aid) {
        return areg.id_to_label(aid);
    };
    if (actions.size() == 1)
    {
        result.action = action_id_label(actions.begin()->first);
    }
    else
    {
        // Stochastic action from along-step!
        std::ostringstream os;
        os << '{'
           << join_stream(
                  actions.begin(),
                  actions.end(),
                  ", ",
                  [&action_id_label, norm](std::ostream& os, auto const& kv) {
                      os << '"' << action_id_label(kv.first)
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
         << repr(this->mfp)
         << ", result.mfp);\n"
            "EXPECT_SOFT_EQ("
         << repr(this->alive) << ", result.alive);\n";
    if (!this->action.empty() && this->action.front() == '{')
        cout << "// ";
    cout << "EXPECT_EQ(" << repr(this->action)
         << ", result.action);\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
/*!
 * Look up the given action and apply it to the state.
 */
void AlongStepTestBase::execute_action(std::string const& label,
                                       CoreState<MemSpace::host>* state)
{
    CELER_EXPECT(state);
    auto const& areg = *this->action_reg();

    auto action_id = areg.find_action(label);
    CELER_VALIDATE(action_id, << "no '" << label << "' action found");
    auto const* step_action = dynamic_cast<CoreStepActionInterface const*>(
        areg.action(action_id).get());
    CELER_VALIDATE(step_action, << "action '" << label << "' cannot execute");
    return this->execute_action(*step_action, state);
}

//---------------------------------------------------------------------------//
/*!
 * Look up the given action and apply it to the state.
 */
void AlongStepTestBase::execute_action(CoreStepActionInterface const& action,
                                       CoreState<MemSpace::host>* state) const
{
    CELER_EXPECT(state);
    CELER_TRY_HANDLE(action.step(*this->core(), *state),
                     LogContextException{this->output_reg().get()});
}

//---------------------------------------------------------------------------//
/*!
 * Inject primaries into the state.
 */
void AlongStepTestBase::extend_from_primaries(Span<Primary const> primaries,
                                              CoreState<MemSpace::host>* state)
{
    CELER_EXPECT(state);

    this->insert_primaries(*state, primaries);
    this->execute_action(*this->primaries_action(), state);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
