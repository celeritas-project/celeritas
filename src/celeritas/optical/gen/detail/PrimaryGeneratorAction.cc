//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/PrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "PrimaryGeneratorAction.hh"

#include <variant>

#include "corecel/Assert.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/KernelLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/CoreTrackData.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/phys/GeneratorRegistry.hh"

#include "PrimaryGeneratorExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
std::shared_ptr<PrimaryGeneratorAction> PrimaryGeneratorAction::make_and_insert(
    ::celeritas::CoreParams const& core_params,
    optical::CoreParams const& params,
    Input&& input)
{
    CELER_EXPECT(input.num_events > 0 && input.primaries_per_event > 0);
    ActionRegistry& actions = *params.action_reg();
    AuxParamsRegistry& aux = *core_params.aux_reg();
    GeneratorRegistry& gen = *params.gen_reg();
    auto result = std::make_shared<PrimaryGeneratorAction>(
        actions.next_id(), aux.next_id(), gen.next_id(), std::move(input));

    actions.insert(result);
    aux.insert(result);
    gen.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, data IDs, and optical properties.
 *
 * \todo Support multiple events and distribution types
 */
PrimaryGeneratorAction::PrimaryGeneratorAction(ActionId id,
                                               AuxId aux_id,
                                               GeneratorId gen_id,
                                               Input inp)
    : action_id_(id), aux_id_(aux_id), gen_id_(gen_id)
{
    CELER_EXPECT(action_id_);
    CELER_EXPECT(aux_id_);
    CELER_EXPECT(gen_id_);

    CELER_VALIDATE(inp.num_events == 1,
                   << "multiple events are not supported for optical primary "
                      "generation");
    CELER_VALIDATE(inp.energy.energy > zero_quantity(),
                   << "expected nonzero energy in optical primary generator");
    CELER_VALIDATE(std::holds_alternative<inp::PointShape>(inp.shape),
                   << "unsupported distribution type for optical primary "
                      "generator position");
    CELER_VALIDATE(std::holds_alternative<inp::IsotropicAngle>(inp.angle),
                   << "unsupported distribution type for optical primary "
                      "generator direction");

    data_.energy = inp.energy.energy;
    data_.position = std::get<inp::PointShape>(inp.shape).pos;
    data_.num_photons = inp.primaries_per_event;

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 */
auto PrimaryGeneratorAction::create_state(MemSpace, StreamId, size_type) const
    -> UPState
{
    return std::make_unique<GeneratorStateBase>();
}

//---------------------------------------------------------------------------//
/*!
 * Description of the action.
 */
std::string_view PrimaryGeneratorAction::description() const
{
    return "generate optical photon primaries";
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void PrimaryGeneratorAction::step(optical::CoreParams const& params,
                                  CoreStateHost& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void PrimaryGeneratorAction::step(optical::CoreParams const& params,
                                  CoreStateDevice& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical photons from distribution data.
 */
template<MemSpace M>
void PrimaryGeneratorAction::step_impl(optical::CoreParams const& params,
                                       optical::CoreState<M>& state) const
{
    CELER_EXPECT(state.aux());

    auto& aux_state = this->counters(*state.aux());
    auto& gen_counters = aux_state.counters;
    auto& counters = state.counters();

    size_type num_gen = min(counters.num_vacancies, gen_counters.num_pending);
    if (num_gen > 0)
    {
        // Generate the optical photons from the distribution data
        this->generate(params, state);

        // Update the optical core state counters
        counters.num_pending -= num_gen;
        counters.num_generated += num_gen;
        counters.num_vacancies -= num_gen;

        // Update the generator counters and statistics
        gen_counters.num_pending -= num_gen;
        gen_counters.num_generated += num_gen;
        aux_state.accum.num_generated += num_gen;
    }
    counters.num_active = state.size() - counters.num_vacancies;
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to generate optical photons.
 */
void PrimaryGeneratorAction::generate(optical::CoreParams const& params,
                                      CoreStateHost& state) const
{
    CELER_EXPECT(state.aux());

    auto const& aux_state = this->counters(*state.aux());
    size_type num_gen
        = min(state.counters().num_vacancies, aux_state.counters.num_pending);

    // Generate optical photons in vacant track slots
    detail::PrimaryGeneratorExecutor execute{
        params.ptr<MemSpace::native>(), state.ptr(), data_, state.counters()};
    celeritas::optical::launch_action(num_gen, execute);
}

//---------------------------------------------------------------------------//
/*!
 * Get generator counters (mutable).
 */
GeneratorStateBase& PrimaryGeneratorAction::counters(AuxStateVec& aux) const
{
    return get<GeneratorStateBase>(aux, aux_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Get generator counters.
 */
GeneratorStateBase const&
PrimaryGeneratorAction::counters(AuxStateVec const& aux) const
{
    return get<GeneratorStateBase>(aux, aux_id_);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void PrimaryGeneratorAction::generate(optical::CoreParams const&,
                                      CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
