//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/PrimaryGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "PrimaryGeneratorAction.hh"

#include <variant>

#include "corecel/Assert.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/io/Logger.hh"
#include "corecel/random/distribution/DistributionInserter.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/KernelLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/CoreTrackData.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/phys/GeneratorRegistry.hh"

#include "detail/PrimaryGeneratorExecutor.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
std::shared_ptr<PrimaryGeneratorAction>
PrimaryGeneratorAction::make_and_insert(CoreParams const& params, Input&& input)
{
    CELER_EXPECT(input);
    ActionRegistry& actions = *params.action_reg();
    AuxParamsRegistry& aux = *params.aux_reg();
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
 * Construct with IDs and distribution.
 */
PrimaryGeneratorAction::PrimaryGeneratorAction(ActionId id,
                                               AuxId aux_id,
                                               GeneratorId gen_id,
                                               Input inp)
    : GeneratorBase(id,
                    aux_id,
                    gen_id,
                    "primary-generate",
                    "generate optical photon primaries")
{
    HostVal<DistributionParamsData> host_params;
    DistributionInserter insert(host_params);

    data_.num_photons = inp.primaries;
    data_.energy = std::visit(insert, inp.energy);
    data_.angle = std::visit(insert, inp.angle);
    data_.shape = std::visit(insert, inp.shape);

    params_ = CollectionMirror<DistributionParamsData>{std::move(host_params)};

    CELER_ENSURE(data_);
    CELER_ENSURE(params_);
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
 * Set the number of pending tracks.
 */
void PrimaryGeneratorAction::insert(optical::CoreStateBase& state) const
{
    if (auto* s = dynamic_cast<CoreStateHost*>(&state))
    {
        return this->insert_impl(*s);
    }
    else if (auto* s = dynamic_cast<CoreStateDevice*>(&state))
    {
        return this->insert_impl(*s);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void PrimaryGeneratorAction::step(CoreParams const& params,
                                  CoreStateHost& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void PrimaryGeneratorAction::step(CoreParams const& params,
                                  CoreStateDevice& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Set the number of pending tracks.
 */
template<MemSpace M>
void PrimaryGeneratorAction::insert_impl(optical::CoreState<M>& state) const
{
    CELER_EXPECT(state.aux());

    auto& aux_state = this->counters(*state.aux());
    aux_state.counters.num_pending = data_.num_photons;
    auto counters = state.sync_get_counters();
    counters.num_pending += data_.num_photons;
    state.sync_put_counters(counters);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical photons from distribution data.
 */
template<MemSpace M>
void PrimaryGeneratorAction::step_impl(CoreParams const& params,
                                       CoreState<M>& state) const
{
    CELER_EXPECT(state.aux());

    auto const& counters = this->counters(*state.aux()).counters;

    if (state.sync_get_counters().num_vacancies > 0 && counters.num_pending > 0)
    {
        // Generate the optical photons from the distribution data
        this->generate(params, state);
    }

    // Update the generator and optical core state counters
    this->update_counters(state);
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to generate optical photons.
 */
void PrimaryGeneratorAction::generate(CoreParams const& params,
                                      CoreStateHost& state) const
{
    CELER_EXPECT(state.aux());

    auto const& aux_state = this->counters(*state.aux());
    size_type num_gen = min(state.sync_get_counters().num_vacancies,
                            aux_state.counters.num_pending);

    // Generate optical photons in vacant track slots
    detail::PrimaryGeneratorExecutor execute{
        params.ptr<MemSpace::native>(), state.ptr(), data_, params_.host_ref()};
    launch_action(num_gen, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void PrimaryGeneratorAction::generate(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
