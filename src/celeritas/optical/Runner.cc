//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Runner.cc
//---------------------------------------------------------------------------//
#include "Runner.hh"

#include <utility>

#include "corecel/io/Logger.hh"
#include "corecel/io/OutputInterfaceAdapter.hh"
#include "corecel/io/OutputRegistry.hh"
#include "corecel/sys/Openmp.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "celeritas/inp/StandaloneInputIO.json.hh"
#include "celeritas/phys/GeneratorRegistry.hh"
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

    ScopedProfiling profile_this{"setup"};
    StreamId stream_id{0};
    auto num_tracks = osi.problem.capacity.tracks;

    // Prepare problem input for json output before it's modified during setup
    auto osi_output = std::make_shared<OutputInterfaceAdapter<Input>>(
        OutputInterface::Category::input, "*", std::make_shared<Input>(osi));

    // Set up the problem from the input
    loaded_ = setup::standalone_input(osi);

    // Save the optical transporter and generator
    CELER_ASSERT(loaded_.problem.transporter);
    CELER_ASSERT(loaded_.problem.generator);
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
        if (CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK)
        {
            CELER_LOG(status) << "Running track-parallel with "
                              << openmp_max_threads() << " max threads";
        }
    }

    // Allocate auxiliary data
    CELER_ASSERT(this->params()->aux_reg());
    state_->aux() = std::make_shared<AuxStateVec>(
        *this->params()->aux_reg(), memspace, stream_id, num_tracks);

    std::move(osi) = {};
}

//---------------------------------------------------------------------------//
/*!
 * Set the number of pending tracks for a primary generator.
 */
void Runner::insert()
{
    auto generate
        = std::dynamic_pointer_cast<optical::PrimaryGeneratorAction const>(
            loaded_.problem.generator);
    CELER_VALIDATE(generate,
                   << "runner call does not match input generator type");

    // Set the number of pending tracks
    generate->insert(*state_);
}

//---------------------------------------------------------------------------//
/*!
 * Insert track initializers.
 */
void Runner::insert(SpanConstTrackInit data)
{
    auto generate
        = std::dynamic_pointer_cast<optical::DirectGeneratorAction const>(
            loaded_.problem.generator);
    CELER_VALIDATE(generate,
                   << "runner call does not match input generator type");

    // Insert track initializers
    generate->insert(*state_, data);
}

//---------------------------------------------------------------------------//
/*!
 * Insert distributions for generating through scintillation or Cherenkov.
 */
void Runner::insert(SpanConstGenDist data)
{
    auto generate = std::dynamic_pointer_cast<optical::GeneratorAction const>(
        loaded_.problem.generator);
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
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical photons and transport to completion.
 */
auto Runner::operator()() const -> Result
{
    ScopedProfiling profile_this{"run"};
    (*loaded_.problem.transporter)(*state_);

    Result result;
    result.counters = state_->accum();
    for (auto gen_id : range(GeneratorId(this->params()->gen_reg()->size())))
    {
        auto const gen = this->params()->gen_reg()->at(gen_id);
        CELER_ASSERT(gen);
        result.counters.generators.push_back(
            gen->counters(*state_->aux()).accum);
    }
    result.action_times
        = loaded_.problem.transporter->get_action_times(*state_->aux());
    result.step_times
        = loaded_.problem.transporter->get_step_times(*state_->aux());

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
