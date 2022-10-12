//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.cc
//---------------------------------------------------------------------------//
#include "Stepper.hh"

#include "corecel/Assert.hh"
#include "corecel/data/Ref.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/track/TrackInitUtils.hh"

#include "ActionRegistry.hh"
#include "CoreParams.hh"
#include "detail/ActionSequence.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with problem parameters and setup options.
 */
template<MemSpace M>
Stepper<M>::Stepper(Input input)
    : params_(std::move(input.params))
    , num_initializers_(input.num_initializers)
{
    CELER_EXPECT(params_);
    CELER_VALIDATE(input.num_track_slots > 0,
                   << "number of track slots has not been set");
    CELER_VALIDATE(input.num_initializers > 0,
                   << "number of initializers has not been set");
    states_ = CollectionStateStore<CoreStateData, M>(params_->host_ref(),
                                                     input.num_track_slots);

    // Create action sequence
    {
        ActionSequence::Options opts;
        opts.sync = input.sync;
        actions_
            = std::make_shared<ActionSequence>(*params_->action_reg(), opts);
    }

    core_ref_.params = get_ref<M>(*params_);
    core_ref_.states = states_.ref();

    CELER_ENSURE(actions_ && *actions_);
}

//---------------------------------------------------------------------------//
//! Default destructor
template<MemSpace M>
Stepper<M>::~Stepper() = default;

//---------------------------------------------------------------------------//
/*!
 * Transport already-initialized states.
 *
 * A single transport step is simply a loop over a toplogically sorted DAG
 * of kernels.
 */
template<MemSpace M>
auto Stepper<M>::operator()() -> result_type
{
    CELER_EXPECT(*this);

    // TODO: refactor initialization
    CELER_VALIDATE(inits_, << "no primaries were given");

    result_type result;

    // Create new tracks from queued primaries or secondaries
    initialize_tracks(core_ref_, &inits_);
    result.active = states_.size() - inits_.vacancies.size();

    actions_->execute(core_ref_);

    // Create track initializers from surviving secondaries
    extend_from_secondaries(core_ref_, &inits_);

    // Get the number of track initializers and active tracks
    result.alive  = states_.size() - inits_.vacancies.size();
    result.queued = inits_.initializers.size();

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize new primaries and transport them for a single step.
 *
 * \todo Currently the track initializers and primary initialization are tied
 * together, so we can only call this once. Once we refactor TrackInitParams we
 * should be able to control the injection of new primaries into the event
 * loop.
 */
template<MemSpace M>
auto Stepper<M>::operator()(VecPrimary primaries) -> result_type
{
    CELER_EXPECT(*this);
    CELER_EXPECT(!primaries.empty());

    CELER_VALIDATE(!inits_,
                   << "primaries were already initialized (currently they "
                      "must be set exactly once, at the first step)");

    // Create track initializers and add primaries
    TrackInitParams::Input inp;
    inp.primaries = std::move(primaries);
    inp.capacity  = num_initializers_;
    TrackInitParams init_params{std::move(inp)};

    // Create track initializers
    resize(&inits_, init_params.host_ref(), states_.size());
    CELER_VALIDATE(init_params.host_ref().primaries.size()
                       <= inits_.initializers.capacity(),
                   << "insufficient initializer capacity ("
                   << inits_.initializers.capacity() << ") for primaries ("
                   << init_params.host_ref().primaries.size() << ')');

    extend_from_primaries(init_params.host_ref(), &inits_);

    return (*this)();
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class Stepper<MemSpace::host>;
template class Stepper<MemSpace::device>;

//---------------------------------------------------------------------------//
} // namespace celeritas
