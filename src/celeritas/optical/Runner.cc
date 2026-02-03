//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Runner.cc
//---------------------------------------------------------------------------//
#include "Runner.hh"

#include <utility>

#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputRegistry.hh"
#include "celeritas/inp/StandaloneInputIO.json.hh"
#include "celeritas/setup/Problem.hh"

#include "CoreParams.hh"
#include "CoreState.hh"
#include "Transporter.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with optical problem input definition.
 */
Runner::Runner(Input&& osi)
{
    CELER_VALIDATE(osi.problem.num_streams == 1,
                   << "standalone optical runner expects a single stream");
    StreamId stream_id{0};
    auto num_tracks = osi.problem.capacity.tracks;

    // Prepare problem input for json output before it's modified during setup
    auto osi_output = std::make_shared<OutputInterfaceAdapter<Input>>(
        OutputInterface::Category::input, "*", std::make_shared<Input>(osi));

    // Set up the problem from the input
    auto loaded = setup::standalone_input(osi);

    // Save the optical transporter and generator
    problem_ = std::move(loaded.problem);
    CELER_ASSERT(problem_.transporter);
    CELER_ASSERT(problem_.generator);
    CELER_ASSERT(stream_id < this->params()->max_streams());

    // Add problem input to output registry
    this->params()->output_reg()->insert(osi_output);

    // Allocate state data
    auto memspace = celeritas::device() ? MemSpace::device : MemSpace::host;
    if (memspace == MemSpace::device)
    {
        state_ = std::make_shared<CoreState<MemSpace::device>>(
            *this->params(), stream_id, num_tracks);
    }
    else
    {
        state_ = std::make_shared<CoreState<MemSpace::host>>(
            *this->params(), stream_id, num_tracks);
    }

    // Allocate auxiliary data
    CELER_ASSERT(this->params()->aux_reg());
    state_->aux() = std::make_shared<AuxStateVec>(
        *this->params()->aux_reg(), memspace, stream_id, num_tracks);

    std::move(osi) = {};
}

//---------------------------------------------------------------------------//
/*!
 * Transport tracks generated with a primary generator.
 */
auto Runner::operator()() -> Result
{
    auto generate
        = std::dynamic_pointer_cast<optical::PrimaryGeneratorAction const>(
            problem_.generator);
    CELER_VALIDATE(generate,
                   << "runner call does not match input generator type");

    // Set the number of pending tracks
    generate->insert(*state_);

    return this->run();
}

//---------------------------------------------------------------------------//
/*!
 * Transport tracks generated directly from track initializers.
 */
auto Runner::operator()(SpanConstTrackInit data) -> Result
{
    auto generate
        = std::dynamic_pointer_cast<optical::DirectGeneratorAction const>(
            problem_.generator);
    CELER_VALIDATE(generate,
                   << "runner call does not match input generator type");

    // Insert track initializers
    generate->insert(*state_, data);

    return this->run();
}

//---------------------------------------------------------------------------//
/*!
 * Transport tracks generated through scintillation or Cherenkov.
 */
auto Runner::operator()(SpanConstGenDist data) -> Result
{
    auto generate = std::dynamic_pointer_cast<optical::GeneratorAction const>(
        problem_.generator);
    CELER_VALIDATE(generate,
                   << "runner call does not match input generator type");
    // Insert optical distributions
    generate->insert(*state_, data);

    /*! \todo We could accumulate and update the number of pending tracks
     * directly in GeneratorAction::insert, though this would be less efficient
     * for some run modes, e.g. offloading distributions through accel where we
     * already know the number of pending tracks.
     */
    auto counters = state_->sync_get_counters();
    for (auto const& d : data)
    {
        counters.num_pending += d.num_photons;
    }
    state_->sync_put_counters(counters);

    return this->run();
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical photons and transport to completion.
 */
auto Runner::run() const -> Result
{
    (*problem_.transporter)(*state_);

    Result result;
    result.counters = state_->accum();
    result.counters.generators.push_back(
        problem_.generator->counters(*state_->aux()).accum);
    result.action_times
        = problem_.transporter->get_action_times(*state_->aux());

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
